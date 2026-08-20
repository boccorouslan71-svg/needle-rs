#!/usr/bin/env python3
"""
Generate tokenizer parity vectors from the SentencePiece model embedded in a
`.cact` blob.

Two references, deliberately:

  * `export.RefTokenizer` — upstream's normative encoder/decoder for the blob
    format, and what `needle-infer::sp_tokenizer` is a port of.
  * `sentencepiece` itself — what the model was actually trained against.

Where the two disagree the reference implementation is wrong about real
SentencePiece, and that is worth knowing rather than silently encoding. The
generator records both and reports any divergence.

Usage (from the repo root):
    PYTHONPATH=needle .venv-parity/bin/python tools/gen_tokenizer_parity.py \
        --cact weights/needle2.cact \
        --sp-model weights/tokenizer.model \
        --output tests/tokenizer_vectors.json
"""

import argparse
import json
import sys

CORPUS = [
    "",
    " ",
    "  ",
    "a",
    "hello",
    "Hello, world!",
    "hello world",
    " leading space",
    "trailing space ",
    "multiple   internal   spaces",
    "\ttab and\nnewline",
    "What's the weather in Paris?",
    "Book a table for 4 at 7pm tomorrow",
    "1234567890",
    "3.14159 and -273.15",
    '{"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}',
    '{"a":1,"b":[2,3],"c":{"d":null,"e":true}}',
    "<tool_call>",
    "</tool_call>",
    "<tool_call>{\"name\": \"search\"}</tool_call>",
    "<|im_start|>user\nhi<|im_end|>",
    "<tools>[{\"name\":\"f\"}]</tools>",
    "<think>reasoning</think>",
    "<tool_result>ok</tool_result>",
    "nested <tool_call> inside <tool_result> text",
    "<tool",
    "<tool_calll>",
    "café naïve résumé",
    "überänderung",
    "你好世界",
    "こんにちは",
    "한국어",
    "مرحبا بالعالم",
    "шифрование",
    "\U0001F600 \U0001F680 \U0001F44D",
    "mixed \U0001F30D unicode café 你好 42",
    "def f(x):\n    return x ** 2\n",
    "SELECT * FROM t WHERE a = 'b';",
    "https://example.com/path?q=1&r=2#frag",
    "snake_case camelCase PascalCase kebab-case",
    "!@#$%^&*()_+-=[]{}|;:'\",.<>/?`~\\",
    "\x00\x01\x02",
    "a" * 200,
    "the quick brown fox jumps over the lazy dog " * 8,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cact", default="weights/needle2.cact")
    ap.add_argument("--sp-model", default="weights/tokenizer.model")
    ap.add_argument("--output", default="tests/tokenizer_vectors.json")
    args = ap.parse_args()

    from needle.model.export import RefTokenizer, read_export, parse_tokenizer_blob

    _, tensors = read_export(args.cact)
    blob = next(t for t in tensors if isinstance(t, (bytes, bytearray)))
    meta = parse_tokenizer_blob(blob)
    ref = RefTokenizer(meta)

    sp = None
    try:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.Load(args.sp_model)
    except Exception as e:  # noqa: BLE001 - informational only
        print(f"[warn] sentencepiece unavailable ({e}); skipping cross-check", file=sys.stderr)

    out = {
        "n_pieces": len(meta["pieces"]),
        "pad_id": meta["pad_id"],
        "eos_id": meta["eos_id"],
        "bos_id": meta["bos_id"],
        "unk_id": meta["unk_id"],
        "add_dummy_prefix": meta["add_dummy_prefix"],
        "byte_fallback": meta["byte_fallback"],
        # Every marker surface with its id, so the Rust side can assert the
        # documented chat-marker ids rather than hardcode them.
        "markers": {p: i for i, (p, t) in enumerate(zip(meta["pieces"], meta["types"])) if t == 3},
        "cases": [],
    }

    divergences = []
    for text in CORPUS:
        ids = ref.encode(text)
        entry = {"text": text, "ids": ids, "decoded": ref.decode(ids)}
        if sp is not None:
            sp_ids = sp.Encode(text, out_type=int)
            entry["sp_ids"] = sp_ids
            if sp_ids != ids:
                divergences.append((text, ids, sp_ids))
        out["cases"].append(entry)

    # Round-trip decode of raw id sequences, including specials.
    id_cases = [
        [meta["bos_id"]],
        [meta["bos_id"], meta["eos_id"]],
        [meta["pad_id"], meta["unk_id"]],
        list(range(0, 32)),
        [i for i in range(100, 140)],
    ]
    out["decode_cases"] = [{"ids": ids, "decoded": ref.decode(ids)} for ids in id_cases]

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False)

    print(
        f"wrote {args.output}: {len(out['cases'])} encode cases, "
        f"{len(out['decode_cases'])} decode cases, {out['n_pieces']} pieces",
        file=sys.stderr,
    )
    if sp is None:
        print("[warn] no sentencepiece cross-check was performed", file=sys.stderr)
    elif divergences:
        print(
            f"[warn] RefTokenizer disagrees with sentencepiece on "
            f"{len(divergences)}/{len(CORPUS)} inputs:",
            file=sys.stderr,
        )
        for text, a, b in divergences[:10]:
            print(f"    {text!r}\n      ref {a}\n      sp  {b}", file=sys.stderr)
    else:
        print(f"[ok] RefTokenizer matches sentencepiece on all {len(CORPUS)} inputs", file=sys.stderr)


if __name__ == "__main__":
    main()
