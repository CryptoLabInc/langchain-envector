from __future__ import annotations

import json

from libs.envector.types import pack_metadata, unpack_metadata


def test_pack_unpack_metadata_roundtrip():
    s = pack_metadata("hello world", {"a": 1})
    obj = unpack_metadata(s)
    assert obj["text"] == "hello world"
    assert obj["metadata"] == {"a": 1}


def test_unpack_metadata_fallback():
    raw = "not-json"
    obj = unpack_metadata(raw)
    assert obj["_raw"] == raw

