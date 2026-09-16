from __future__ import annotations

import threading
from typing import Any, Optional, Tuple

from .config import EnvectorConfig

# pyenvector holds one connection per process: `Index.init_connect` replaces it
# and closes the previous channel. Stores that ask for the endpoint already open
# reuse it; anything different reconnects.
_ACTIVE_CONNECTION: Optional[Tuple[Tuple[Any, ...], Any]] = None
_CONNECTION_LOCK = threading.Lock()


def _connection_key(config: EnvectorConfig) -> Tuple[Any, ...]:
    c = config.connection
    return (
        c.address,
        c.host,
        c.port,
        c.access_token,
        c.secure,
        c.kms_address,
        c.kms_secure,
        c.kms_ca_cert,
    )


def _reusable_indexer(key: Tuple[Any, ...]):
    """Return the live indexer opened for `key`, or None to connect afresh."""
    global _ACTIVE_CONNECTION
    if _ACTIVE_CONNECTION is None:
        return None
    active_key, indexer = _ACTIVE_CONNECTION
    if active_key != key:
        return None

    from pyenvector.index import Index

    # Only reuse the recorded indexer while it is still the process-global one
    # and its channel is open; anything else means someone reconnected under us.
    if Index._default_indexer is not indexer:
        _ACTIVE_CONNECTION = None
        return None
    try:
        if not indexer.is_connected():
            _ACTIVE_CONNECTION = None
            return None
    except Exception:
        _ACTIVE_CONNECTION = None
        return None
    return indexer


class EnvectorClient:
    """Thin convenience client around the high-level `pyenvector` SDK.

    - Establishes a connection (reusing the process-wide one when the endpoint
      matches, so several stores can coexist)
    - Initializes key and index configuration
    - Optionally creates the index if missing
    - Provides access to the envector `Index` instance
    """

    def __init__(self, config: EnvectorConfig):
        self.config = config
        self._ev = None
        self._index = None

    def _connect(self, ev_client) -> None:
        c = self.config.connection
        if not c.address and not (c.host and c.port):
            raise ValueError("Either address or host+port must be provided.")

        key = _connection_key(self.config)
        with _CONNECTION_LOCK:
            self._connect_locked(ev_client, key)

    def _connect_locked(self, ev_client, key: Tuple[Any, ...]) -> None:
        global _ACTIVE_CONNECTION

        c = self.config.connection
        reused = _reusable_indexer(key)
        if reused is not None:
            # Adopt the open channel instead of tearing it down and rebuilding it.
            ev_client.indexer = reused
        elif c.address:
            ev_client.init_connect(
                address=c.address, access_token=c.access_token, secure=c.secure
            )
        else:
            ev_client.init_connect(
                host=c.host, port=c.port, access_token=c.access_token, secure=c.secure
            )

        # Optional KMS connection (shares the indexer's auth session when
        # called after the indexer is in place)
        if c.kms_address:
            ev_client.init_kms_connect(
                kms_address=c.kms_address,
                secure=c.kms_secure,
                ca_cert=c.kms_ca_cert,
            )

        _ACTIVE_CONNECTION = (key, ev_client.indexer)

    def _resolve_index_params(self) -> Optional[dict]:
        """Fold `index_type` and `index_params` into the one dict the SDK reads.

        The SDK ignores `index_type` as soon as `index_params` is present, and
        raises a bare ``KeyError('index_type')`` if that dict has no
        ``index_type`` key — so fill it in from `index_type` rather than letting
        a half-specified pair fail deep inside the SDK. The dict is copied
        because the SDK rewrites it in place (upper-casing the type and adding
        IVF defaults), which would otherwise mutate the caller's config object.
        """
        i = self.config.index
        if i.index_params is None:
            return {"index_type": i.index_type} if i.index_type else None
        params = dict(i.index_params)
        if "index_type" not in params:
            if not i.index_type:
                raise ValueError(
                    "IndexSettings.index_params must carry an 'index_type' key, "
                    "or IndexSettings.index_type must be set."
                )
            params["index_type"] = i.index_type
        return params

    def init(self):
        import pyenvector as ev

        c = self.config.connection
        k = self.config.key
        i = self.config.index

        ev_client = ev.EnvectorClient()
        self._connect(ev_client)

        index_params = self._resolve_index_params()
        if not c.kms_address and not k.key_path:
            raise ValueError(
                "KeyConfig.key_path is required unless ConnectionConfig.kms_address is set."
            )

        # Index config + key setup (KMS-managed keys require key_path=None)
        try:
            ev_client.init_index_config(
                index_name=i.index_name,
                dim=i.dim,
                key_path=None if c.kms_address else k.key_path,
                key_id=k.key_id,
                seal_mode=k.seal_mode,
                seal_kek_path=k.seal_kek_path,
                preset=k.preset,
                eval_mode=k.eval_mode,
                query_encryption=i.query_encryption,
                index_encryption=i.index_encryption,
                index_params=index_params,
                auto_key_setup=True,
                description=i.description,
                metadata_encryption=i.metadata_encryption,
            )
        except ValueError as e:
            if "does not match the default key path" not in str(e):
                raise
            # pyenvector pins one key path per process; say so in our terms.
            raise ValueError(
                f"pyenvector keeps one key path per process; this store asked for "
                f"{k.key_path!r} while another key path is already active. "
                "Use one KeyConfig.key_path per process."
            ) from e

        # Create index if missing. create_index already returns the bound Index,
        # so reuse it instead of re-opening the same index by name.
        index = None
        if self.config.create_if_missing:
            idx_list = ev_client.get_index_list()
            if i.index_name not in idx_list:
                index = ev_client.create_index(
                    index_name=i.index_name,
                    dim=i.dim,
                    # a fresh copy: init_index_config rewrote the first one in place
                    index_params=self._resolve_index_params(),
                )

        # Bind index instance
        self._index = index if index is not None else ev.Index(i.index_name)
        self._ev = ev_client
        return self

    @property
    def index(self):
        if self._index is None:
            raise RuntimeError("Client not initialized. Call init().")
        return self._index

    @property
    def ev(self):
        if self._ev is None:
            raise RuntimeError("Client not initialized. Call init().")
        return self._ev
