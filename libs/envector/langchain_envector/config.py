from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConnectionConfig:
    address: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    access_token: Optional[str] = None
    secure: Optional[bool] = None  # None = SDK default (secure iff a token is provided)
    # Optional enVector KMS connection (pyenvector >= 1.5.0).
    kms_address: Optional[str] = None  # host:port of the KMS combined service
    kms_secure: Optional[bool] = None  # independent from the endpoint `secure` option
    kms_ca_cert: Optional[str] = None  # PEM CA bundle path (or PEM bytes)


@dataclass
class KeyConfig:
    key_path: Optional[str] = None  # local key dir; omit when keys are KMS-managed
    key_id: Optional[str] = None
    preset: Optional[str] = None
    eval_mode: Optional[str] = None
    seal_mode: Optional[str] = None
    seal_kek_path: Optional[str] = None


@dataclass
class IndexSettings:
    index_name: str
    dim: int
    query_encryption: str = "plain"
    index_encryption: str = "cipher"
    index_type: str = "FLAT"  # FLAT | IVF_FLAT | IVF_VCT (case-insensitive)
    output_fields: List[str] = field(default_factory=lambda: ["metadata"])
    fetch_k: Optional[int] = None  # over-fetch to support client-side filters
    description: Optional[str] = None
    metadata_encryption: Optional[bool] = None
    index_params: Optional[dict] = None  # catch-all for any additional index parameters


@dataclass
class WriteSettings:
    """Whether each write path waits for the server before returning.

    enVector writes are asynchronous server-side: the call returns once the
    request is accepted, and the server finishes the work afterwards. What that
    means for the next read differs per operation, so each default below was
    measured against a live stack rather than assumed.

    - insert: rows become searchable through ``Index.insert(load=True)``, which
      the SDK does by default. Waiting additionally blocks until the shards are
      merged and saved — durability, not visibility — and cost a flat ~14s per
      call regardless of batch size, while not waiting never lost a row across
      batches of 2, 100 and 400. So it stays off.
    - delete: the SDK's own default is to wait, and the wait returned
      immediately in measurement. Left on.
    - update: the call returns at swap-commit with the rebuilt rows' visibility
      still pending. Without the wait the updated row was missing from the very
      next search every time. Left on. Updates additionally wait for any
      un-awaited inserts to merge first — see `Envector._drain_pending_inserts`.

    Set any of these to ``False`` for fire-and-forget bulk work, then wait once
    at the end.

    Only the waiting is configured here. The SDK's per-call tuning knobs
    (``execute_until``, ``n_workers``, ``use_row_insert``, ...) keep their own
    defaults and reach ``Index.insert`` through ``add_texts(**kwargs)``.
    """

    await_insert: bool = False
    await_delete: bool = True
    await_update: bool = True

    # Shared polling budget for the await_* waits above.
    timeout_s: float = 600.0
    poll_interval_s: float = 1.0

    # Budget for draining un-awaited inserts before an update/upsert. Separate
    # and much larger than `timeout_s` because that wait grows with the number
    # of un-awaited insert batches — the server merges them one at a time, so
    # 20 batches took ~125s — and timing it out only forces a retry of the same
    # wait. Nothing is mutated until the drain succeeds.
    drain_timeout_s: float = 3600.0


@dataclass
class EnvectorConfig:
    connection: ConnectionConfig
    key: KeyConfig
    index: IndexSettings
    create_if_missing: bool = True
    write: WriteSettings = field(default_factory=WriteSettings)
