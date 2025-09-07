from __future__ import annotations

from typing import Optional

from .config import EnvectorConfig


class EnvectorClient:
    """Thin convenience client around the high-level `es2` SDK.

    - Establishes a connection
    - Initializes key and index configuration
    - Optionally creates the index if missing
    - Provides access to the ES2 `Index` instance
    """

    def __init__(self, config: EnvectorConfig):
        self.config = config
        self._es2 = None
        self._index = None

    def init(self):
        import es2

        c = self.config.connection
        k = self.config.key
        i = self.config.index

        es2_client = es2.ES2()

        # Connection
        if c.address:
            es2_client.init_connect(address=c.address, access_token=c.access_token)
        else:
            if not (c.host and c.port):
                raise ValueError("Either address or host+port must be provided.")
            es2_client.init_connect(host=c.host, port=c.port, access_token=c.access_token)

        # Key path baseline for Index
        from es2.index import Index as _Index

        _Index.init_key_path(k.key_path)

        # Index config + key setup
        es2_client.init_index_config(
            index_name=i.index_name,
            dim=i.dim,
            key_path=k.key_path,
            key_id=k.key_id,
            seal_mode=k.seal_mode,
            seal_kek_path=k.seal_kek_path,
            preset=k.preset,
            eval_mode=k.eval_mode,
            query_encryption=i.query_encryption,
            index_encryption="cipher",  # server vectors are always encrypted
            index_type=i.index_type,
            auto_key_setup=True,
        )

        # Create index if missing
        if self.config.create_if_missing:
            idx_list = es2_client.get_index_list()
            if i.index_name not in idx_list:
                es2_client.create_index(index_name=i.index_name, dim=i.dim)

        # Bind index instance
        self._index = es2.Index(i.index_name)
        self._es2 = es2_client
        return self

    @property
    def index(self):
        if self._index is None:
            raise RuntimeError("Client not initialized. Call init().")
        return self._index

    @property
    def es2(self):
        if self._es2 is None:
            raise RuntimeError("Client not initialized. Call init().")
        return self._es2

