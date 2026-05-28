from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConnectionConfig:
    address: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    access_token: Optional[str] = None
    secure: bool = True  # default to secure connection if not specified


@dataclass
class KeyConfig:
    key_path: str
    key_id: str
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
    index_type: str = "flat"
    output_fields: List[str] = field(default_factory=lambda: ["metadata"])
    fetch_k: Optional[int] = None  # over-fetch to support client-side filters
    description: Optional[str] = None
    metadata_encryption: Optional[str] = None
    index_params: Optional[dict] = None  # catch-all for any additional index parameters


@dataclass
class EnvectorConfig:
    connection: ConnectionConfig
    key: KeyConfig
    index: IndexSettings
    create_if_missing: bool = True
