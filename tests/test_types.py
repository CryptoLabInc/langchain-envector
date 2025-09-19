from __future__ import annotations

import json

from langchain_envector.types import pack_metadata, unpack_metadata


def test_pack_unpack_metadata_roundtrip():
    s = pack_metadata("hello world", {"a": 1})
    obj = unpack_metadata(s)
    assert obj["text"] == "hello world"
    assert obj["metadata"] == {"a": 1}


def test_unpack_metadata_fallback():
    raw = "not-json"
    obj = unpack_metadata(raw)
    assert obj["_raw"] == raw


def test_unpack_metadata_accepts_dict():
    payload = {"text": "foo", "metadata": {"x": 1}}
    obj = unpack_metadata(payload)
    assert obj is payload


def test_unpack_metadata_handles_single_item_list():
    payload = pack_metadata("hi", {"a": 2})
    obj = unpack_metadata([payload])
    assert obj["text"] == "hi"
    assert obj["metadata"] == {"a": 2}


def test_unpack_metadata_parses_python_literal_dict():
    raw = str({"text": "bar", "metadata": {"y": 3}})
    obj = unpack_metadata(raw)
    assert obj["text"] == "bar"
    assert obj["metadata"] == {"y": 3}
