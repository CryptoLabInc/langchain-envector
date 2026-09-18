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

    enVector accepts a write first and completes it server-side afterwards.
    Inserted rows are searchable as soon as the call returns, so
    ``await_insert`` is off by default and only adds durability. Updates return
    before the rebuilt rows are visible and deletes before the shards are
    rebuilt, so those wait by default. Set a flag to ``False`` for
    fire-and-forget bulk work and wait once at the end.

    Only the waiting is configured here; the SDK's per-call knobs
    (``execute_until``, ``n_workers``, ``use_row_insert``, ...) reach
    ``Index.insert`` through ``add_texts(**kwargs)``.
    """

    await_insert: bool = False
    await_delete: bool = True
    await_update: bool = True

    # Polling budget for the delete/update/upsert waits. Inserts keep the
    # SDK's own budget (a day) unless `timeout_s` is passed to add_texts.
    timeout_s: float = 600.0
    poll_interval_s: float = 1.0

    # Budget for waiting on un-awaited inserts before an update/upsert. Larger
    # than `timeout_s` because it grows with the number of pending batches.
    drain_timeout_s: float = 3600.0


@dataclass
class EnvectorConfig:
    connection: ConnectionConfig
    key: KeyConfig
    index: IndexSettings
    create_if_missing: bool = True
    write: WriteSettings = field(default_factory=WriteSettings)
