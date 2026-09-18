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


def _init_with_fake_sdk(key=None, ev_client=None, **index_kwargs) -> dict:
    """Run init() against a stub SDK and return what it was told about the index."""
    import sys
    import types

    from langchain_envector import client as client_mod

    ev_client = ev_client or _FakeEvClient()
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
            key=key or KeyConfig(key_path="./keys", key_id="kid"),
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


def test_missing_key_path_without_kms_is_a_clear_error():
    # KeyConfig fields became optional for KMS-managed keys; without a KMS
    # address the SDK would fail deep inside key setup instead.
    try:
        _init_with_fake_sdk(key=KeyConfig(key_id="kid"))
        assert False, "Expected a ValueError naming key_path"
    except ValueError as e:
        assert "key_path" in str(e) and "kms_address" in str(e)


def test_second_key_path_in_one_process_is_explained_in_our_terms():
    class _PinnedEvClient(_FakeEvClient):
        def init_index_config(self, **kwargs):
            raise ValueError(
                "Key path ./keys_b does not match the default key path ./keys_a. "
                "Please reinitialize. pyenvector.init()"
            )

    try:
        _init_with_fake_sdk(
            key=KeyConfig(key_path="./keys_b", key_id="kid"),
            ev_client=_PinnedEvClient(),
        )
        assert False, "Expected a ValueError about one key path per process"
    except ValueError as e:
        assert "one key path per process" in str(e) and "./keys_b" in str(e)


# ---------------------------------------------------------------------------
# Connection reuse (client.py). These run without the SDK: the stub stands in
# for pyenvector and pyenvector.index, which is what _reusable_indexer imports.
# ---------------------------------------------------------------------------


class _StubIndexer:
    def __init__(self) -> None:
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected


class _StubIndex:
    """pyenvector.index.Index as far as connection reuse can see it."""

    _default_indexer = None


class _ConnectingEvClient(_FakeEvClient):
    def __init__(self) -> None:
        super().__init__()
        self.indexer = None
        self.connects: list = []

    def init_connect(self, **kwargs) -> None:
        self.connects.append(kwargs)
        self.indexer = _StubIndexer()
        _StubIndex._default_indexer = self.indexer


def _with_stub_sdk(index_cls, run):
    """Install the stub pyenvector modules, reset the reuse cache, run `run`."""
    import sys
    import types

    from langchain_envector import client as client_mod

    fake = types.ModuleType("pyenvector")
    fake.Index = lambda name: object()
    fake_index = types.ModuleType("pyenvector.index")
    fake_index.Index = index_cls
    fake.index = fake_index

    saved = {name: sys.modules.get(name) for name in ("pyenvector", "pyenvector.index")}
    prev_conn = client_mod._ACTIVE_CONNECTION
    sys.modules["pyenvector"] = fake
    sys.modules["pyenvector.index"] = fake_index
    client_mod._ACTIVE_CONNECTION = None
    _StubIndex._default_indexer = None
    try:
        return run(fake)
    finally:
        client_mod._ACTIVE_CONNECTION = prev_conn
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _store_client(fake, address: str, ev_client) -> EnvectorClient:
    fake.EnvectorClient = lambda: ev_client
    cfg = EnvectorConfig(
        connection=ConnectionConfig(address=address),
        key=KeyConfig(key_path="./keys", key_id="kid"),
        index=IndexSettings(index_name="idx", dim=32),
        create_if_missing=False,
    )
    return EnvectorClient(cfg).init()


def test_second_store_on_the_same_endpoint_reuses_the_connection():
    def run(fake):
        a, b = _ConnectingEvClient(), _ConnectingEvClient()
        _store_client(fake, "host:1", a)
        _store_client(fake, "host:1", b)
        assert len(a.connects) == 1
        assert b.connects == []  # adopted a's channel instead of reconnecting
        assert b.indexer is a.indexer

    _with_stub_sdk(_StubIndex, run)


def test_a_different_endpoint_connects_afresh():
    def run(fake):
        a, b = _ConnectingEvClient(), _ConnectingEvClient()
        _store_client(fake, "host:1", a)
        _store_client(fake, "host:2", b)
        assert len(b.connects) == 1
        assert b.indexer is not a.indexer

    _with_stub_sdk(_StubIndex, run)


def test_a_closed_channel_is_not_reused():
    def run(fake):
        a, b = _ConnectingEvClient(), _ConnectingEvClient()
        _store_client(fake, "host:1", a)
        a.indexer.connected = False
        _store_client(fake, "host:1", b)
        assert len(b.connects) == 1

    _with_stub_sdk(_StubIndex, run)


def test_sdk_without_the_private_indexer_attribute_fails_loudly():
    # Reuse leans on pyenvector's private Index._default_indexer. If the SDK
    # drops it, the second store must not quietly fall back to reconnecting —
    # that is the behaviour that used to close the first store's channel.
    class _IndexWithoutPrivate:
        pass

    def run(fake):
        a, b = _ConnectingEvClient(), _ConnectingEvClient()
        _store_client(fake, "host:1", a)  # nothing cached yet: no import needed
        try:
            _store_client(fake, "host:1", b)
            assert False, "Expected a RuntimeError about Index._default_indexer"
        except RuntimeError as e:
            assert "_default_indexer" in str(e)

    _with_stub_sdk(_IndexWithoutPrivate, run)
