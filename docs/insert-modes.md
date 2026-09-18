# Insert modes: when to pass `use_row_insert`

EnVector accepts an insert on one of two paths, and `add_texts` / `add_documents` let you pick:

- **bulk** (the default) packs the whole call into one ciphertext block. The block's size is set
  by the index dimension, not by how many documents you are adding, so one call costs about the
  same whether it carries one document or a thousand.
- **row** (`use_row_insert=True`) encrypts one small ciphertext per document, and the server adds
  them one document at a time. There is no fixed block to pay for, but the work grows with the
  number of documents in the call.

EnVector applies the row path only to calls carrying fewer documents than the index dimension.
Asking for it with more raises a `UserWarning`, and those documents go in on the bulk path.

## Short answer

**Leave `use_row_insert` off.** It does not make the call return sooner, and it does not make the
documents searchable sooner — not even for a single document, where the two paths come out within
about 10% of each other. From two documents up it is 1.5× to 5× slower, and it writes roughly
twice as much to server storage on every call.

The one thing it buys is that the index **finishes merging** sooner after a very small call. That
is worth something only if you wait for the merge: you pass `await_completion=True`, or you call
`update_documents` / `upsert_documents` soon after inserting, which waits for that store's pending
inserts to merge first. In that case:

| index dim | row finishes merging sooner for a call of up to |
|---|---|
| 256 | 4 documents |
| 384 | 4 documents |
| 512 | 3 documents |
| 768 | 2 documents |
| 1024 | 2 documents |
| 1536 | no size — use bulk |

```python
# default: adding documents, however few, to an index that is being searched
store.add_documents(new_docs)

# only when you wait for the merge and the call is small, at a low dimension
store.add_documents(new_docs, use_row_insert=True, await_completion=True)
```

## The measurements

Time for `add_texts` to return, against an index already holding 8192 documents. Median of three
calls; each call's merge was allowed to finish before the next one started.

| index dim | bulk, any size | row, 1 doc | row, 2 | row, 3 | row, 4 | row, 6 | row, 8 |
|---|---|---|---|---|---|---|---|
| 256 | 1.3 s | 1.4 s | 2.1 s | 2.6 s | 3.4 s | 4.8 s | 6.1 s |
| 384 | 1.8 s | 2.0 s | 3.1 s | 4.1 s | 4.9 s | 6.6 s | 9.4 s |
| 512 | 2.3 s | 2.7 s | 3.4 s | 5.2 s | 6.2 s | 9.2 s | 11.9 s |
| 768 | 3.1 s | 3.2 s | 5.4 s | 7.6 s | 9.4 s | 12.4 s | — |
| 1024 | 4.0 s | 4.3 s | 6.4 s | 8.7 s | 10.2 s | 14.8 s | 19.4 s |
| 1536 | 5.9 s | 6.9 s | 10.2 s | 13.0 s | 15.0 s | 20.8 s | 29.1 s |

The bulk column is one number because the path is flat in the number of documents: across ten
sizes from 1 to 32 documents, every dimension stayed inside a 0.4 s band.

Documents become searchable a fraction of a second after either call returns, so the picture for
"when can I find it" is the same as the table above.

Time until the index has finished merging, which is the one place the row path is ahead:

| index dim | 1 doc, bulk → row | 2 docs | 3 docs | 4 docs |
|---|---|---|---|---|
| 256 | 12.4 → 6.5 s | 12.3 → 8.1 s | 11.3 → 9.0 s | 13.5 → 10.2 s |
| 512 | 12.9 → 9.5 s | 14.1 → 10.3 s | 14.6 → 11.7 s | 14.6 → 13.1 s |
| 1024 | 16.4 → 12.2 s | 16.9 → 14.3 s | 17.0 → 17.2 s | 16.9 → 18.1 s |
| 1536 | 17.2 → 14.9 s | 18.3 → 18.4 s | 17.8 → 21.5 s | 17.7 → 23.4 s |

Storage written per call, which the row path loses at every size:

| index dim | bulk | row |
|---|---|---|
| 256 | 7.9 MB | 16.9 MB |
| 512 | 15.8 MB | 33.7 MB |
| 1024 | 31.5 MB | 67.2 MB |
| 1536 | 47.3 MB | 100.8 MB |

Measured on a single-node EnVector deployment, pyenvector 1.6.2, preset ip3, eval mode mms32,
FLAT index, with nothing else running against the server. Your numbers will differ with hardware,
index size and load, and the row path in particular degrades further when calls arrive faster than
the server merges them. The shape — bulk flat in the number of documents, row rising with it — is
a property of the two paths, so if the boundary matters to your deployment, time both at the size
you actually add.
