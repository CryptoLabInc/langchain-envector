# TODO — deferred work

Items intentionally left out of the pyenvector 1.6.0 compatibility pass
(`feat/update-1.6`). Each one is independent of 1.6 compatibility: the
integration works against 1.6.0 without them.

## Multi-key support (`pyenvector.key.Key`)

pyenvector 1.6 added a `Key` handle with explicit per-index binding
(envector-msa #2098). `EnvectorClient.create_index(key=...)` and
`Index(index_name, index_config, key=...)` accept it, `register_key` /
`generate_and_register_key` produce it, and the eval key is verified against the
server's by SHA256. `langchain_envector` still uses the legacy single-key path:
one `key_path` + `key_id` per client, derived from `KeyConfig`.

To support it, `KeyConfig` would need to carry (or accept) a `Key` handle, and
`EnvectorClient.init` would need to branch between the key-path flow and the
key-handle flow — the latter skips `init_index_config`'s on-disk key guards
entirely (`use_key_stream=True`).

## Cloud / remote key stores

`init_index_config` and `create_index` accept `key_store` (`aws` / `gcp` /
`vault`), plus `region_name`, `bucket_name`, `secret_prefix`, `vault_addr` and
`vault_mount`; `KeyProtectionStrategy`, `AWSKeyStoreConfig`,
`GCPKeyStoreConfig` and `VaultKeyStoreConfig` are exported from `pyenvector`.
`KeyConfig` exposes none of these, so keys must live on the local filesystem (or
be KMS-managed via `ConnectionConfig.kms_address`). Requires the
`pyenvector[aws]` / `pyenvector[gcp]` extras.

## Pre-encrypted metadata (`SealedBlob`)

`Index.insert` accepts `SealedBlob`-wrapped ciphertext to store metadata the
client encrypted itself (e.g. via `KMSClient.encrypt_metadata`), skipping SDK
re-encryption. `add_texts` always packs plaintext into the
`{"text": ..., "metadata": ...}` JSON envelope and lets the SDK encrypt it.

## `get_by_ids`

`VectorStoreIntegrationTests.has_get_by_ids` stays `False`. The only
metadata-fetch API is `Indexer.get_metadata`, which addresses rows by
`(shard_idx, row_idx)` positions rather than by `item_id`, so there is no way to
resolve an `item_id` back to a document without a search.

## `add_texts(ids=...)` is still ignored

enVector issues its own `item_id` values and cannot insert at a caller-chosen
ID: `Index.upsert` routes an ID-bearing item to the update arm, and an ID
matching no live row is reported in `not_found_item_ids` instead of being
inserted. The LangChain standard tests that require caller-chosen IDs to survive
`add_documents` therefore remain `xfail`
(`tests/integration_tests/test_vectorstore.py`). `upsert_documents` covers the
half that 1.6 does support: overwriting items by their returned IDs.

`add_texts` also keeps returning `List[int]` (the native `item_id` type) rather
than the `List[str]` LangChain's type hints declare. Callers who need strings
get them for free from `Document.id`, which pydantic coerces, and `delete`
accepts both.


Standalone repro scripts for the two upstream items below, with the exact
environment they were measured on, live outside this repo in
`../lc-envector-audit-minseok/`.

## Upstream: search races the shard bookkeeping when an index is emptied

Reproduced against a 1.6-era stack with the raw SDK, no LangChain involved.
After deleting every row of an index, a search over it *sometimes* raises
`InternalError: ... code = NotFound desc = shard list for index <name> is empty`
instead of returning the empty result a never-populated index returns. It is a
race whose rate varies between runs: 5 of 8 rounds raised on one run, 2 of 8
on another.

Measured scope, so the workaround stays as narrow as the problem:

- Partial deletes are consistent — repeated "delete one row, search
  immediately" rounds returned exactly the remaining row count every time.
- Deleting everything and inserting again then searching was correct in 6 of 6
  runs; the error did not appear over live data.
- `delete(await_completion=True)` does not close the window: the wait returns
  before the emptied index settles.

`Envector._similarity_search_with_scores` normalises that error to `[]`, but
only after `_index_is_empty()` confirms with the server that `row_count == 0`.
The message alone is not enough to act on — reporting "no matches" for a
transient error over live data would look exactly like data loss. If the server
cannot answer, the original error is re-raised.

Drop `_is_empty_shard_list_error` / `_index_is_empty` and their unit tests once
the backend answers a search over an emptied index the same way it answers one
over a never-populated index.

Repro:

```python
idx = ev.create_index(name, 32)
ids = idx.insert([v0, v1], metadata=["a", "b"], await_completion=True)
idx.load()
idx.delete(ids, await_completion=True)
idx.search(v0, top_k=2, output_fields=["metadata"])  # ~60%: InternalError, expected []
```

## Upstream: mutating a row whose insert has not merged drops it from search

The most serious of the three, because it is silent. `Index.insert` defaults to
not waiting for the server-side merge; updating or upserting one of those rows
before the merge completes removes it from search results while
`not_found_item_ids` stays empty, `row_count` still counts it, and no exception
is raised. Measured over 6 rounds each: 4/6 lost with a mixed upsert, 3/6 with a
plain update, 0/6 when the insert was awaited first. delete over unmerged rows
was correct 8/8, and search alone is unaffected.

`Envector._drain_pending_inserts` works around it: `add_texts` keeps the request
ids of inserts it did not wait for, and any update/upsert waits for those to
reach the merged stage before mutating. Inserts stay at ~0.13s and the ~15s
merge wait is paid once, only when rows are actually mutated. Remove the drain
once the server either folds the pending merge into the mutation or refuses the
mutation loudly.

## Upstream: one connection per process

`EnvectorClient.init_connect` delegates to the `Index.init_connect`
**classmethod**, which disconnects and replaces the process-global
`Index._default_indexer`. Connecting a second time closes the channel every
earlier client is still holding, and those clients then fail every call with
`ValueError: Cannot invoke RPC on closed channel!`.

`langchain_envector.client` works around it by recording what the live
connection was opened with and adopting it whenever a new store asks for exactly
the same endpoint (`tests/integration_tests/test_multi_store.py`). The
workaround only covers stores that share one endpoint. Two stores pointing at
**different** enVector servers in one process still cannot coexist — the second
`init_connect` closes the first one's channel — and there is nothing the
integration can do about that from outside the SDK.

A fix upstream would be to stop treating the indexer as class state, or to let
`init_connect` return a connection the caller owns.
