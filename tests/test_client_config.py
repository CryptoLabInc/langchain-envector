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


class _FakeEvClient:
    def __init__(self):
        self.indexer = object()
        self.index_config = None

    def init_connect(self, **kwargs):
        pass

    def init_index_config(self, **kwargs):
        self.index_config = kwargs


def _init_with_fake_sdk(**index_kwargs) -> dict:
    """Run init() against a stub SDK and return what it was told about the index.

    No pytest fixtures here: `scripts/run_unit_tests.py` calls test functions
    with no arguments, so the stubbing is undone by hand instead.
    """
    import sys
    import types

    from langchain_envector import client as client_mod

    ev_client = _FakeEvClient()
    fake = types.ModuleType("pyenvector")
    fake.EnvectorClient = lambda: ev_client
    fake.Index = lambda name: object()

    prev_mod = sys.modules.get("pyenvector")
    prev_conn = client_mod._ACTIVE_CONNECTION
    sys.modules["pyenvector"] = fake
    # A stale global connection would make _connect adopt someone else's indexer.
    client_mod._ACTIVE_CONNECTION = None
    try:
        cfg = EnvectorConfig(
            connection=ConnectionConfig(address="dummy:0"),
            key=KeyConfig(key_path="./keys", key_id="kid"),
            index=IndexSettings(index_name="idx", dim=32, **index_kwargs),
            create_if_missing=False,
        )
        EnvectorClient(cfg).init()
    finally:
        client_mod._ACTIVE_CONNECTION = prev_conn
        if prev_mod is None:
            del sys.modules["pyenvector"]
        else:
            sys.modules["pyenvector"] = prev_mod
    return ev_client.index_config


def test_index_encryption_reaches_the_sdk():
    # Regression: this was hardcoded to "cipher", so a caller asking for a
    # plaintext index silently got an encrypted one. The server supports both.
    sent = _init_with_fake_sdk(index_encryption="plain")
    assert sent["index_encryption"] == "plain"


def test_index_encryption_still_defaults_to_cipher():
    sent = _init_with_fake_sdk()
    assert sent["index_encryption"] == "cipher"
