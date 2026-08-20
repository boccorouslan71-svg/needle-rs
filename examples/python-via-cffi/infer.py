#!/usr/bin/env python3
"""
Run Needle inference from Python through the needle-rs C ABI (ctypes).

Both model versions are supported, and the version is inferred from the model
path: a `.cact` file is Needle v2, anything else is v1. No Python ML
dependencies — standard library and ctypes only.

    # Needle v2 — one file, tokenizer included
    python infer.py --model ../../weights/needle2.cact \
        --query "What's the weather in Paris?" --constrain

    # Needle v2 — confidence gating (v1 has retrieval, but no confidence head)
    python infer.py --model ../../weights/needle2.cact --analyse

    # Needle v1 — weights plus a separate vocabulary
    python infer.py --model ../../weights/needle.safetensors \
        --vocab ../../weights/vocab.txt

Calls libneedle_c.so (Linux) / libneedle_c.dylib (macOS) / needle_c.dll
(Windows). Build it with: cargo build --release -p needle-c
"""

import argparse
import ctypes
import json
import os
import platform
import sys

STREAM_CB = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_void_p)

DEFAULT_TOOLS = json.dumps([
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    },
])


def declare(lib):
    """Declare both surfaces. `needle_free_str` and `needle_last_error` are shared."""
    c_void, c_str = ctypes.c_void_p, ctypes.c_char_p

    # --- shared
    # Anything the C side allocates is declared as c_void_p, not c_char_p:
    # ctypes converts a c_char_p result into a Python bytes object and discards
    # the original pointer, so the pointer handed to needle_free_str would be
    # Python-owned memory. That aborts. Keep the raw pointer, read through a
    # cast, free the pointer.
    lib.needle_free_str.restype = None
    lib.needle_free_str.argtypes = [c_void]
    lib.needle_last_error.restype = c_str
    lib.needle_last_error.argtypes = []

    # --- v1
    lib.needle_load.restype = c_void
    lib.needle_load.argtypes = [c_str, c_str]
    lib.needle_run.restype = c_void
    lib.needle_run.argtypes = [c_void, c_str, c_str]
    lib.needle_run_stream.restype = c_void
    lib.needle_run_stream.argtypes = [c_void, c_str, c_str, STREAM_CB, c_void]
    lib.needle_free.restype = None
    lib.needle_free.argtypes = [c_void]

    # --- v2
    lib.needle_v2_load.restype = c_void
    lib.needle_v2_load.argtypes = [c_str]
    lib.needle_v2_run.restype = c_void
    lib.needle_v2_run.argtypes = [c_void, c_str, c_str]
    lib.needle_v2_run_json.restype = c_void
    lib.needle_v2_run_json.argtypes = [c_void, c_str, c_str]
    lib.needle_v2_generate.restype = c_void
    lib.needle_v2_generate.argtypes = [
        c_void, c_str, c_str, ctypes.c_size_t, ctypes.c_float, ctypes.c_uint64, ctypes.c_int
    ]
    lib.needle_v2_run_stream.restype = c_void
    lib.needle_v2_run_stream.argtypes = [c_void, c_str, c_str, STREAM_CB, c_void]
    lib.needle_v2_confidence.restype = ctypes.c_bool
    lib.needle_v2_confidence.argtypes = [c_void, c_str, ctypes.POINTER(ctypes.c_float)]
    lib.needle_v2_confidence_for.restype = ctypes.c_bool
    lib.needle_v2_confidence_for.argtypes = [
        c_void, c_str, c_str, c_str, ctypes.POINTER(ctypes.c_float)
    ]
    lib.needle_v2_contrastive_dim.restype = ctypes.c_size_t
    lib.needle_v2_contrastive_dim.argtypes = [c_void]
    lib.needle_v2_retrieve_tools.restype = ctypes.c_size_t
    lib.needle_v2_retrieve_tools.argtypes = [
        c_void, c_str, ctypes.POINTER(c_str), ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_float),
    ]
    lib.needle_v2_free.restype = None
    lib.needle_v2_free.argtypes = [c_void]
    return lib


def last_error(lib):
    err = lib.needle_last_error()
    return err.decode() if err else "(no error reported)"


def take(lib, ptr):
    """Copy a returned string and free the original — the C side transfers ownership."""
    if not ptr:
        return None
    text = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", "replace")
    lib.needle_free_str(ptr)
    return text


def default_lib_path():
    release = os.path.join(os.path.dirname(__file__), "..", "..", "target", "release")
    name = {"Linux": "libneedle_c.so", "Darwin": "libneedle_c.dylib"}.get(
        platform.system(), "needle_c.dll"
    )
    return os.path.join(release, name)


def run_v2(lib, args):
    handle = lib.needle_v2_load(args.model.encode())
    if not handle:
        sys.exit(f"failed to load {args.model}: {last_error(lib)}")
    try:
        print(f"── Needle v2  ·  {os.path.basename(args.model)} (tokenizer included)")
        q, t = args.query.encode(), args.tools.encode()

        streamed = None
        if args.stream:
            def on_token(_id, piece, _ud):
                sys.stdout.write(piece.decode("utf-8", "replace"))
                sys.stdout.flush()
            cb = STREAM_CB(on_token)
            text = take(lib, lib.needle_v2_run_stream(handle, q, t, cb, None))
            print()
            streamed = text
        elif args.constrain or args.max_tokens != 128 or args.temperature != 0.0:
            text = take(lib, lib.needle_v2_generate(
                handle, q, t, args.max_tokens, args.temperature, args.seed,
                1 if args.constrain else 0,
            ))
        else:
            text = take(lib, lib.needle_v2_run(handle, q, t))

        if text is None:
            sys.exit(f"inference failed: {last_error(lib)}")
        # Already on screen if it was streamed.
        if streamed is None:
            print(text)

        payload = take(lib, lib.needle_v2_run_json(handle, q, t))
        if payload:
            print(f"\npayload: {payload}")

        if args.analyse:
            analyse_v2(lib, handle, args, text)
    finally:
        lib.needle_v2_free(handle)


def analyse_v2(lib, handle, args, completion):
    """The two probe heads. v1 has no equivalent for confidence."""
    print("\n── probe heads")

    # The confidence head scores a judgement, not a question: it wants the
    # prompt plus the completion. needle_v2_confidence_for assembles that.
    # Passing the bare query to needle_v2_confidence reads near zero however
    # good the answer is, which is why this uses the _for variant.
    conf = ctypes.c_float()
    ok = lib.needle_v2_confidence_for(
        handle, args.query.encode(), args.tools.encode(),
        (completion or "").encode(), ctypes.byref(conf),
    )
    if ok:
        verdict = "trust it" if conf.value >= 0.5 else "escalate"
        print(f"confidence: {conf.value * 100:.1f}%  → {verdict}")
    else:
        print(f"confidence: unavailable — {last_error(lib)}")

    dim = lib.needle_v2_contrastive_dim(handle)
    if dim == 0:
        print("retrieval: no contrastive head in this container")
        return

    tools = json.loads(args.tools)
    descs = [f"{t['name']}: {t.get('description', '')}".strip() for t in tools]
    arr = (ctypes.c_char_p * len(descs))(*[d.encode() for d in descs])
    idx = (ctypes.c_size_t * len(descs))()
    scores = (ctypes.c_float * len(descs))()
    n = lib.needle_v2_retrieve_tools(
        handle, args.query.encode(), arr, len(descs), len(descs), idx, scores
    )
    print(f"retrieval ({dim}-d):")
    for i in range(n):
        print(f"  {scores[i]:+.4f}  {descs[idx[i]]}")


def run_v1(lib, args):
    if not args.vocab or not os.path.exists(args.vocab):
        sys.exit("Needle v1 needs --vocab (it keeps the tokenizer outside the weights)")
    handle = lib.needle_load(args.model.encode(), args.vocab.encode())
    if not handle:
        sys.exit(f"failed to load {args.model}: {last_error(lib)}")
    try:
        print(f"── Needle v1  ·  {os.path.basename(args.model)} + {os.path.basename(args.vocab)}")
        for flag in ("constrain", "analyse"):
            if getattr(args, flag):
                print(f"note: --{flag} is v2 only; ignoring it for v1", file=sys.stderr)

        q, t = args.query.encode(), args.tools.encode()
        if args.stream:
            def on_token(_id, piece, _ud):
                sys.stdout.write(piece.decode("utf-8", "replace"))
                sys.stdout.flush()
            cb = STREAM_CB(on_token)
            print("streamed: ", end="", flush=True)
            text = take(lib, lib.needle_run_stream(handle, q, t, cb, None))
            print()
        else:
            text = take(lib, lib.needle_run(handle, q, t))

        if text is None:
            sys.exit(f"inference failed: {last_error(lib)}")
        # v1 post-processes: the <tool_call> marker is stripped and the caller's
        # original tool-name casing restored, so the streamed tokens are a
        # progress view and this is the answer. They will differ.
        print(f"{'result:   ' if args.stream else ''}{text}")
    finally:
        lib.needle_free(handle)


def main():
    ap = argparse.ArgumentParser(description="Needle inference via the C ABI")
    ap.add_argument("--lib", default=default_lib_path())
    ap.add_argument("--model", default="../../weights/needle2.cact",
                    help="a .cact container (v2) or .safetensors weights (v1)")
    ap.add_argument("--vocab", default="../../weights/vocab.txt", help="v1 only")
    ap.add_argument("--query", default="What's the weather in Paris?")
    ap.add_argument("--tools", default=DEFAULT_TOOLS)
    ap.add_argument("--stream", action="store_true", help="print tokens as generated")
    ap.add_argument("--constrain", action="store_true", help="v2: restrict to the schema")
    ap.add_argument("--analyse", action="store_true", help="v2: run the probe heads")
    ap.add_argument("--max-tokens", type=int, default=128, help="v2 only")
    ap.add_argument("--temperature", type=float, default=0.0, help="v2 only; 0 is greedy")
    ap.add_argument("--seed", type=int, default=0, help="v2 only")
    args = ap.parse_args()

    if not os.path.exists(args.lib):
        sys.exit(f"library not found: {args.lib}\nbuild it: cargo build --release -p needle-c")
    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")

    lib = declare(ctypes.CDLL(args.lib))
    # The container format decides the version; there is no flag to get wrong.
    if args.model.endswith(".cact"):
        run_v2(lib, args)
    else:
        run_v1(lib, args)


if __name__ == "__main__":
    main()
