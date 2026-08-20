#!/usr/bin/env python3
"""
Generate end-to-end parity vectors for the v2 path: prompt in, exact generated
token ids out.

The v2 counterpart of `tools/gen_e2e_vectors.py`. Where the forward-pass ladder
(`gen_v2_forward_parity.py`) pins intermediates for one prompt, this pins the
*whole pipeline* — chat template, tokenizer, prefill, greedy decode, detokenise —
across a spread of cases, and is the gate invasive changes are checked against.

Deliberate coverage:
  * a prompt longer than the 256-token attention window, so anything that
    mis-sizes or mis-indexes the KV cache fails here
  * a case that emits <think> before the call
  * a case with no applicable tool, where the model should decline
  * several tool counts, so the prompt length varies across chunk boundaries

Usage (from the repo root):
    PYTHONPATH=needle:tools JAX_PLATFORMS=cpu .venv-parity/bin/python \
        tools/gen_v2_e2e_parity.py
"""

import argparse
import json
import sys

WEATHER = '{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}'
EMAIL = '{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]}}'
FLIGHT = '{"name":"book_flight","description":"Book a flight between two airports","parameters":{"type":"object","properties":{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"}},"required":["origin","destination"]}}'
MUSIC = '{"name":"play_music","description":"Play a song or playlist","parameters":{"type":"object","properties":{"track":{"type":"string"}},"required":["track"]}}'
TIMER = '{"name":"set_timer","description":"Set a countdown timer","parameters":{"type":"object","properties":{"seconds":{"type":"integer"},"label":{"type":"string"}},"required":["seconds"]}}'
CALC = '{"name":"calculate","description":"Evaluate an arithmetic expression","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}'
ODD = '{"name":"qq_weather_lookup","description":"Get current weather for a city","parameters":{"type":"object","properties":{"wibble_place":{"type":"string"}},"required":["wibble_place"]}}'


def tools(*defs):
    return "[" + ",".join(defs) + "]"


CASES = [
    ("What's the weather in Paris?", tools(WEATHER)),
    ("Is it raining in Tokyo right now?", tools(WEATHER)),
    ("Email alice@example.com saying the build is green", tools(WEATHER, EMAIL)),
    ("Book me a flight from LHR to JFK on Friday", tools(WEATHER, EMAIL, FLIGHT)),
    ("Play Kind of Blue", tools(MUSIC, TIMER)),
    ("Set a timer for 10 minutes called pasta", tools(MUSIC, TIMER)),
    ("What is 17 * 23 + 4?", tools(CALC)),
    ("Weather in Reykjavik and then email bob@example.com about it",
     tools(WEATHER, EMAIL, FLIGHT, MUSIC)),
    # Unguessable tool name: exercises the tokenizer on rare pieces.
    ("What's the weather in Lisbon?", tools(ODD)),
    # No applicable tool: the model should decline rather than invent a call.
    ("Write me a poem about the sea", tools(WEATHER, TIMER)),
    # Every tool at once: the longest prompt, comfortably past the 256-token
    # attention window, so a mis-sized or mis-indexed KV cache fails here.
    ("I need the weather in Oslo, then book a flight there from Berlin next "
     "Tuesday, email the itinerary to travel@example.com, put on some jazz, "
     "and set a timer for twenty minutes so I remember to check in.",
     tools(WEATHER, EMAIL, FLIGHT, MUSIC, TIMER, CALC, ODD)),
    ("Compute 99 / 3 and then tell me the weather in Cairo",
     tools(CALC, WEATHER, EMAIL, FLIGHT, MUSIC, TIMER)),
    ("", tools(WEATHER)),
    ("weather", tools(WEATHER)),
]

MAX_NEW = 96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cact", default="weights/needle2.cact")
    ap.add_argument("--output", default="tests/v2_e2e_vectors.json")
    args = ap.parse_args()

    import jax.numpy as jnp
    from cact_params import load_cact_params
    from needle.model.export import parse_tokenizer_blob, RefTokenizer
    from needle.model import decode as D
    from needle.model.architecture import precompute_rope_freqs
    from needle.model.tokenizer import (
        IM_START, IM_END, TOOLS_START, TOOLS_END, BOS_ID, EOS_ID,
    )

    params, config, meta = load_cact_params(args.cact)
    params = D._f32(params)
    tok = RefTokenizer(parse_tokenizer_blob(meta["tokenizer_blob"]))

    out = {
        "kv_window": int(config.kv_window or 0),
        "max_new_tokens": MAX_NEW,
        "cases": [],
    }

    for query, tools_json in CASES:
        prompt = (IM_START + "user\n" + TOOLS_START + tools_json + TOOLS_END + "\n"
                  + query + IM_END + "\n" + IM_START + "assistant\n")
        prompt_ids = ([BOS_ID] + tok.encode(prompt))[: config.max_seq_len - 1]
        max_len = min(config.max_seq_len, len(prompt_ids) + MAX_NEW)

        head_dim = (config.attn_dim or config.d_model) // config.num_heads
        cos, sin = precompute_rope_freqs(head_dim, max_len, config.rope_theta)
        dcfg = D.decode_cfg(config, kv_window=config.kv_window)
        kc, vc = D.init_kv_cache(config, 1, max_len)
        hist = jnp.zeros((1, max_len), jnp.int32).at[0, : len(prompt_ids)].set(
            jnp.asarray(prompt_ids, jnp.int32))
        valid = jnp.ones((1, max_len), bool)
        sink = jnp.zeros((1, max_len), bool) if config.kv_window else None

        logits, kc, vc = D.forward_cached(
            params, dcfg, jnp.asarray([prompt_ids], jnp.int32), kc, vc,
            jnp.asarray(0, jnp.int32), cos, sin, None, False, hist, valid, sink)
        nxt = int(jnp.argmax(logits[0, -1]))

        generated, pos = [], len(prompt_ids)
        # Greedy, stopping on EOS or the assistant-turn end, exactly as the Rust
        # engine does.
        im_end = tok.p2id.get(IM_END)
        while len(generated) < MAX_NEW and pos < max_len:
            if nxt == EOS_ID or nxt == im_end:
                break
            generated.append(nxt)
            hist = hist.at[0, pos].set(nxt)
            logits, kc, vc = D.forward_cached(
                params, dcfg, jnp.asarray([[nxt]], jnp.int32), kc, vc,
                jnp.asarray(pos, jnp.int32), cos, sin, None, False, hist, valid, sink)
            nxt = int(jnp.argmax(logits[0, -1]))
            pos += 1

        text = tok.decode(generated)
        out["cases"].append({
            "query": query,
            "tools": tools_json,
            "prompt_tokens": len(prompt_ids),
            "generated_ids": generated,
            "text": text,
        })
        print(f"  {len(prompt_ids):5d} prompt + {len(generated):3d} gen  "
              f"{query[:38]!r} -> {text[:52]!r}", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    longest = max(c["prompt_tokens"] for c in out["cases"])
    print(f"wrote {args.output}: {len(out['cases'])} cases, longest prompt "
          f"{longest} tokens (window {out['kv_window']})", file=sys.stderr)
    if longest <= out["kv_window"]:
        print("[warn] no case exceeds the attention window; KV-cache bugs "
              "would not be caught", file=sys.stderr)


if __name__ == "__main__":
    main()
