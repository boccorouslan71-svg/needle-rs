//! Parity for the embedded SentencePiece tokenizer against upstream.
//!
//! Requires (both gitignored / generated):
//!   weights/needle2.cact          — `huggingface.co/Cactus-Compute/needle2`
//!   tests/tokenizer_vectors.json  — `tools/gen_tokenizer_parity.py`
//!
//! The fixture generator cross-checks upstream's `RefTokenizer` against real
//! `sentencepiece` before writing, so matching it here means matching the
//! tokenizer the model was trained with — not just matching a reference port.
//!
//! Run: cargo test -p needle-infer --test tokenizer_v2_parity -- --nocapture

use needle_infer::cact::Cact;
use needle_infer::sp_tokenizer::{SpTokenizer, CHAT_MARKERS};

const CACT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../weights/needle2.cact");
const VECTORS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/tokenizer_vectors.json");

/// Special-token ids `needle/model/tokenizer.py` documents for v2.
const EXPECTED_MARKER_IDS: [u32; 10] = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

fn fixtures() -> Option<(SpTokenizer, serde_json::Value)> {
    if !std::path::Path::new(CACT).exists() || !std::path::Path::new(VECTORS).exists() {
        eprintln!(
            "skipping tokenizer parity: need weights/needle2.cact and \
             tests/tokenizer_vectors.json\n  see docs/v2-port-record.md"
        );
        return None;
    }
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(VECTORS).expect("read vectors"))
            .expect("parse vectors");
    let cact = Cact::load(CACT).expect("load cact");
    let idx = cact.layout().expect("layout").tokenizer.expect("embedded tokenizer");
    let blob = cact.raw_tensor(idx).expect("raw tokenizer");
    Some((SpTokenizer::from_blob(blob).expect("decode tokenizer"), v))
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

#[test]
fn header_and_piece_table_match_reference() {
    let Some((t, v)) = fixtures() else { return };
    assert_eq!(t.vocab_size(), v["n_pieces"].as_u64().unwrap() as usize);
    assert_eq!(t.pad_id, v["pad_id"].as_u64().unwrap() as u32);
    assert_eq!(t.eos_id, v["eos_id"].as_u64().unwrap() as u32);
    assert_eq!(t.bos_id, v["bos_id"].as_u64().unwrap() as u32);
    assert_eq!(t.unk_id, v["unk_id"].as_u64().unwrap() as u32);
    assert_eq!(t.add_dummy_prefix, v["add_dummy_prefix"].as_bool().unwrap());
    assert_eq!(t.byte_fallback, v["byte_fallback"].as_bool().unwrap());

    // Every user-defined marker must resolve to the same id.
    let markers = v["markers"].as_object().unwrap();
    assert!(!markers.is_empty());
    for (surface, id) in markers {
        assert_eq!(
            t.id_of(surface),
            Some(id.as_u64().unwrap() as u32),
            "marker {surface:?}"
        );
    }
}

/// The v2 special-token ids differ from v1's; pin them so a checkpoint that
/// renumbers them fails loudly instead of emitting the wrong control tokens.
#[test]
fn chat_marker_ids_match_documented_values() {
    let Some((t, _)) = fixtures() else { return };
    let got = t.chat_marker_ids().expect("all ten chat markers present");
    assert_eq!(got, EXPECTED_MARKER_IDS.to_vec(), "markers: {CHAT_MARKERS:?}");
    assert_eq!((t.pad_id, t.eos_id, t.bos_id, t.unk_id), (0, 1, 2, 3));
}

#[test]
fn encode_matches_reference_exactly() {
    let Some((t, v)) = fixtures() else { return };
    let cases = v["cases"].as_array().unwrap();
    assert!(cases.len() >= 40, "expected a broad corpus, got {}", cases.len());
    for c in cases {
        let text = c["text"].as_str().unwrap();
        let want = ids(&c["ids"]);
        let got = t.encode(text);
        assert_eq!(got, want, "encode {text:?}");

        // The fixture also carries real sentencepiece output where available.
        if let Some(sp) = c.get("sp_ids") {
            assert_eq!(got, ids(sp), "encode {text:?} vs sentencepiece");
        }
    }
}

#[test]
fn decode_matches_reference_exactly() {
    let Some((t, v)) = fixtures() else { return };
    for c in v["cases"].as_array().unwrap() {
        let want = c["decoded"].as_str().unwrap();
        let got = t.decode(&ids(&c["ids"]));
        assert_eq!(got, want, "decode of {:?}", c["text"].as_str().unwrap());
    }
    for c in v["decode_cases"].as_array().unwrap() {
        let seq = ids(&c["ids"]);
        assert_eq!(t.decode(&seq), c["decoded"].as_str().unwrap(), "decode {seq:?}");
    }
}

/// Encoding must never emit an id outside the piece table — the tied LM head
/// indexes the embedding with these directly.
#[test]
fn every_encoded_id_is_in_range() {
    let Some((t, v)) = fixtures() else { return };
    for c in v["cases"].as_array().unwrap() {
        for id in t.encode(c["text"].as_str().unwrap()) {
            assert!(
                (id as usize) < t.vocab_size(),
                "id {id} out of range for {:?}",
                c["text"].as_str().unwrap()
            );
        }
    }
}

/// Byte fallback should make every byte string representable, so arbitrary
/// input round-trips rather than collapsing to `<unk>`.
#[test]
fn arbitrary_text_round_trips() {
    let Some((t, _)) = fixtures() else { return };
    assert!(t.byte_fallback, "needle2 ships byte_fallback");
    let samples = [
        "plain ascii",
        "tabs\tand\nnewlines",
        "\u{1F9EE} \u{1F4CE} \u{2728}",
        "ᚠᚢᚦᚨᚱᚲ",
        "\u{200B}zero width\u{200B}",
        "a\u{0301}combining",
        "{\"k\": \"\u{00FC}ber\"}",
    ];
    for s in samples {
        let round = t.decode(&t.encode(s));
        assert_eq!(round, s, "round trip {s:?}");
    }
}
