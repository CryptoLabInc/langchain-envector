"""Unit tests for the index-parameter resolution in EnvectorClient.

These do not touch pyenvector: `_resolve_index_params` is the piece that decides
what the SDK is told about the index type, and getting it wrong is silent — the
SDK ignores `index_type` whenever `index_params` is present.
"""

from __future__ import annotations

from langchain_envector.client import EnvectorClient
from langchain_envector.config import (
    ConnectionConfig,
    EnvectorConfig,
    IndexSettings,
    KeyConfig,
)


def _client(**index_kwargs) -> EnvectorClient:
    cfg = EnvectorConfig(
        connection=ConnectionConfig(address="dummy:0"),
        key=KeyConfig(key_path="./keys", key_id="kid"),
        index=IndexSettings(index_name="idx", dim=32, **index_kwargs),
    )
    return EnvectorClient(cfg)


def test_index_type_becomes_index_params():
    assert _client()._resolve_index_params() == {"index_type": "FLAT"}


def test_index_params_are_not_mutated():
    # The SDK rewrites the dict it is given in place (upper-casing the type and
    # filling in IVF defaults). Passing the caller's dict through would rewrite
    # their IndexSettings; hand the SDK a copy instead.
    user_params = {"index_type": "ivf_flat", "nlist": 8}
    client = _client(index_params=user_params)

    resolved = client._resolve_index_params()
    resolved["index_type"] = "MUTATED"
    resolved["extra"] = 1

    assert user_params == {"index_type": "ivf_flat", "nlist": 8}
    assert client.config.index.index_params is user_params


def test_partial_index_params_take_index_type_from_settings():
    # index_params without an index_type key makes the SDK raise a bare
    # KeyError('index_type'), and index_type would have been ignored anyway.
    client = _client(index_type="IVF_FLAT", index_params={"nlist": 8})
    assert client._resolve_index_params() == {"index_type": "IVF_FLAT", "nlist": 8}


def test_index_params_index_type_wins_over_settings():
    client = _client(index_type="FLAT", index_params={"index_type": "IVF_VCT"})
    assert client._resolve_index_params()["index_type"] == "IVF_VCT"


def test_missing_index_type_everywhere_is_a_clear_error():
    client = _client(index_type="", index_params={"nlist": 8})
    try:
        client._resolve_index_params()
        assert False, "Expected a ValueError naming the missing index_type"
    except ValueError as e:
        assert "index_type" in str(e)


def test_resolution_is_independent_per_call():
    # init() resolves twice (config + create); the second call must not see
    # whatever the SDK did to the first result.
    client = _client(index_params={"index_type": "flat"})
    first = client._resolve_index_params()
    first["index_type"] = "IVF_VCT"
    assert client._resolve_index_params() == {"index_type": "flat"}
