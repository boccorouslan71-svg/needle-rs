/**
 * WASM end-to-end test for both Needle versions (Node.js).
 *
 * The browser demo loads either version from the same wasm module and adapts its
 * UI to what that version supports. This asserts the contract that adapter
 * relies on: which methods exist, what they return, and where the two differ.
 * A broken binding is caught here rather than in a deployed page.
 *
 * Prerequisites:
 *   wasm-pack build crates/needle-wasm --target nodejs --release --out-dir ../../pkg-nodejs/
 *   weights/needle2.cact                    (huggingface.co/Cactus-Compute/needle2)
 *   weights/needle.safetensors + vocab.txt  (huggingface.co/Abdalrahman/needle-rs-safetensors)
 *
 * Run from the workspace root:
 *   node crates/needle-wasm/tests/node_e2e_v2.js
 *
 * Either version is skipped if its files are absent. Exit code 0 = pass.
 */

"use strict";

const fs = require("fs");
const path = require("path");

const PKG = path.resolve(__dirname, "../../../pkg-nodejs/needle_wasm.js");
const CACT = path.resolve(__dirname, "../../../weights/needle2.cact");
const V1_WEIGHTS = path.resolve(__dirname, "../../../weights/needle.safetensors");
const V1_VOCAB = path.resolve(__dirname, "../../../weights/vocab.txt");

const TOOLS = JSON.stringify([
  {
    name: "get_weather",
    description: "Get current weather for a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" }, unit: { type: "string" } },
      required: ["city"],
    },
  },
  {
    name: "send_email",
    description: "Send an email to a recipient",
    parameters: {
      type: "object",
      properties: { to: { type: "string" }, body: { type: "string" } },
      required: ["to", "body"],
    },
  },
]);

let passed = 0;
const failures = [];

function check(name, cond, detail) {
  if (cond) {
    passed++;
    console.log(`  ok   ${name}`);
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ""}`);
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

function extractBetween(text, open, close) {
  const i = text.indexOf(open);
  if (i < 0) return null;
  const rest = text.slice(i + open.length);
  const j = rest.indexOf(close);
  return j < 0 ? rest : rest.slice(0, j);
}

/** The demo's parser: v2 wraps the payload in markers, v1 emits it bare. */
function payloadOf(text) {
  const call = extractBetween(text, "<tool_call>", "</tool_call>");
  return (call !== null ? call : text).trim();
}

function testV2(mod) {
  const { NeedleV2Wasm } = mod;
  const cact = new Uint8Array(fs.readFileSync(CACT));

  console.log(`\n── Needle v2 — container ${(cact.length / 1e6).toFixed(2)} MB`);

  const engine = NeedleV2Wasm.load(cact);
  check("v2 load returns an engine", engine !== undefined && engine !== null);
  if (!engine) return;

  // --- run / run_json
  const text = engine.run("What's the weather in Paris?", TOOLS);
  check("v2 run returns text", typeof text === "string" && text.length > 0, text);
  check("v2 run mentions the tool", text.includes("get_weather"), text.slice(0, 80));

  const payload = engine.run_json("What's the weather in Paris?", TOOLS);
  check("v2 run_json returns a JSON array", payload.trim().startsWith("["), payload);
  let parsed = null;
  try {
    parsed = JSON.parse(payload);
  } catch (e) {
    /* reported below */
  }
  check("v2 run_json payload parses", Array.isArray(parsed), payload);
  check(
    "v2 payload names a declared tool",
    Array.isArray(parsed) && parsed.every((c) => ["get_weather", "send_email"].includes(c.name)),
    payload
  );

  // --- generate, greedy and determinism
  const a = engine.generate("Weather in Berlin?", TOOLS, 48, 0, 0, false);
  const b = engine.generate("Weather in Berlin?", TOOLS, 48, 0, 0, false);
  check("v2 greedy generate is deterministic", a === b);

  const constrained = engine.generate("Weather in Berlin?", TOOLS, 48, 0, 0, true);
  check("v2 constrained generate produces a call", constrained.includes("get_weather"), constrained.slice(0, 80));

  const s1 = engine.generate("Weather in Berlin?", TOOLS, 32, 0.8, 7, false);
  const s2 = engine.generate("Weather in Berlin?", TOOLS, 32, 0.8, 7, false);
  check("v2 sampling is seed-deterministic", s1 === s2);

  // --- streaming reassembles the same text
  let streamed = "";
  let calls = 0;
  const streamResult = engine.run_stream("What's the weather in Rome?", TOOLS, (_id, piece) => {
    calls++;
    streamed += piece;
  });
  check("v2 run_stream fired the callback", calls > 0, `${calls} calls`);
  check("v2 streamed pieces rebuild the text", streamed === streamResult);

  // --- reasoning trace, as the demo parses it
  const reason = engine.generate(
    "Weather in Reykjavik and then email bob@example.com about it",
    TOOLS,
    96,
    0,
    0,
    false
  );
  const think = extractBetween(reason, "<think>", "</think>");
  check("v2 reasoning trace is parseable when emitted", think === null || think.length > 0);

  // --- probe heads
  const dim = engine.contrastive_dim();
  check("v2 contrastive_dim is positive", dim > 0, String(dim));

  const emb = engine.encode_contrastive("get_weather: Get current weather for a city");
  check("v2 encode_contrastive returns a vector", emb && emb.length === dim, emb ? String(emb.length) : "null");
  if (emb) {
    const norm = Math.sqrt(emb.reduce((s, v) => s + v * v, 0));
    check("v2 embedding is unit norm", Math.abs(norm - 1) < 1e-3, String(norm));
  }

  const logit = engine.confidence("What's the weather in Paris?");
  check("v2 confidence returns a finite logit", typeof logit === "number" && Number.isFinite(logit), String(logit));

  // The head scores a completion, not a query. confidence_for assembles the
  // prompt+completion input it was trained on; a correct call must outrank a
  // mismatched one, and the bare query must be uninformative.
  const q = "What's the weather in Paris?";
  const out = engine.run(q, TOOLS);
  const pRight = engine.confidence_for(q, TOOLS, out);
  const pWrong = engine.confidence_for(q, TOOLS,
    '<tool_call>[{"name":"send_email","arguments":{"to":"bob"}}]</tool_call>');
  const pBare = 1 / (1 + Math.exp(-logit));
  check("v2 confidence_for returns a probability", pRight > 0 && pRight < 1, String(pRight));
  check("v2 confidence_for rates its own call above 50%", pRight > 0.5, pRight.toFixed(3));
  check("v2 confidence_for rates a mismatched call below 10%", pWrong < 0.1, pWrong.toFixed(3));
  check("v2 confidence_for ranks right above wrong", pRight > pWrong, `${pRight.toFixed(3)} > ${pWrong.toFixed(3)}`);
  check("v2 bare-query confidence is uninformative", pBare < 0.05, pBare.toFixed(4));

  const descs = JSON.stringify([
    "get_weather: Get current weather for a city",
    "send_email: Send an email to a recipient",
    "play_music: Play a song or playlist",
  ]);
  const ranked = JSON.parse(engine.retrieve_tools("what is the weather in Paris", descs, 3));
  check("v2 retrieve_tools returns ranked pairs", Array.isArray(ranked) && ranked.length === 3, JSON.stringify(ranked));
  check("v2 retrieval is sorted descending", ranked.every((r, i) => i === 0 || ranked[i - 1][1] >= r[1]), JSON.stringify(ranked));
  check("v2 weather query ranks get_weather first", ranked[0][0] === 0, JSON.stringify(ranked));

  // --- malformed input must not throw across the boundary
  const bad = engine.retrieve_tools("q", "not json", 2);
  check("v2 bad retrieval JSON degrades to []", bad === "[]", bad);

  // --- the demo's payload parser must find a call
  check("v2 payload parses out of the markers", payloadOf(text).startsWith("["), payloadOf(text));
}

function testV1(mod) {
  const { NeedleWasm } = mod;
  const weights = new Uint8Array(fs.readFileSync(V1_WEIGHTS));
  const vocab = fs.readFileSync(V1_VOCAB, "utf8");

  console.log(`\n── Needle v1 — weights ${(weights.length / 1e6).toFixed(2)} MB + vocab ${(vocab.length / 1e3).toFixed(0)} KB`);

  const engine = NeedleWasm.load(weights, vocab);
  check("v1 load returns an engine", engine !== undefined && engine !== null);
  if (!engine) return;

  const text = engine.run("What's the weather in Paris?", TOOLS);
  check("v1 run returns text", typeof text === "string" && text.length > 0, text);

  // v1 emits the payload bare, with no <tool_call> markers.
  check("v1 emits no tool_call markers", !text.includes("<tool_call>"), text.slice(0, 80));
  const payload = payloadOf(text);
  check("v1 payload parses as JSON", (() => { try { JSON.parse(payload); return true; } catch { return false; } })(), payload);

  let streamed = "";
  let calls = 0;
  const streamResult = engine.run_stream("Weather in Rome?", TOOLS, (_id, piece) => {
    calls++;
    streamed += piece;
  });
  check("v1 run_stream fired the callback", calls > 0, `${calls} calls`);
  // v1 post-processes: the returned text has the <tool_call> marker stripped and
  // the caller's original tool-name casing restored, so the streamed pieces are a
  // progress view rather than the answer. The demo settles on the returned text
  // when streaming ends; this pins that contract so a change is noticed.
  check(
    "v1 streams the raw marker but returns it stripped",
    streamed.includes("<tool_call>") && !streamResult.includes("<tool_call>"),
    `streamed=${JSON.stringify(streamed.slice(0, 40))} returned=${JSON.stringify(streamResult.slice(0, 40))}`
  );

  // Name restoration: a camelCase tool comes back camelCase even though the
  // model emits snake_case.
  const camel = JSON.stringify([
    {
      name: "getWeather",
      description: "Get current weather for a city",
      parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    },
  ]);
  const restored = engine.run("What is the weather in Paris?", camel);
  check("v1 restores the caller's tool-name casing", restored.includes("getWeather"), restored);

  // The demo gates these on capability flags, so the contract matters.
  check("v1 has no confidence head", typeof engine.confidence !== "function");
  check("v1 has no generate/temperature path", typeof engine.generate !== "function");

  const dim = engine.contrastive_dim();
  check("v1 contrastive_dim is reported", Number.isInteger(dim), String(dim));
  if (dim > 0) {
    const descs = JSON.stringify([
      "get_weather: Get current weather for a city",
      "send_email: Send an email to a recipient",
    ]);
    const ranked = JSON.parse(engine.retrieve_tools("weather in Paris", descs, 2));
    check("v1 retrieve_tools returns ranked pairs", Array.isArray(ranked) && ranked.length === 2, JSON.stringify(ranked));
  }
}

function main() {
  if (!fs.existsSync(PKG)) {
    console.log(`skipping wasm e2e: pkg-nodejs not found at ${PKG}`);
    return 0;
  }
  const mod = require(PKG);

  let ran = 0;
  if (fs.existsSync(CACT)) { testV2(mod); ran++; }
  else console.log(`\nskipping v2: ${CACT} not found`);

  if (fs.existsSync(V1_WEIGHTS) && fs.existsSync(V1_VOCAB)) { testV1(mod); ran++; }
  else console.log(`\nskipping v1: ${V1_WEIGHTS} not found`);

  if (ran === 0) {
    console.log("\nno model files present — nothing to test");
    return 0;
  }

  console.log(`\n${passed} passed, ${failures.length} failed (${ran}/2 versions exercised)`);
  if (failures.length) {
    console.log("\nfailures:");
    for (const f of failures) console.log(`  - ${f}`);
    return 1;
  }
  return 0;
}

process.exit(main());
