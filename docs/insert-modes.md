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

**If your client and your EnVector server are on the same fast network, leave `use_row_insert`
off.** It does not make the call return sooner and does not make the documents searchable sooner —
not even for a single document, where the two paths come out within about 10% of each other. From
two documents up it is 1.5× to 5× slower.

**If your client reaches the server over a slow link, turn it on for small calls.** The bulk path
uploads a block whose size comes from the index dimension, not from how many documents you are
adding: at dim 1024 that is 31.5 MB per call, whether it carries one document or a thousand. The
row path uploads 61.5 KB per document. Below about 100 Mbps that difference decides the question on
its own — see "What each path uploads".

Whichever you use, the documents end up occupying the same space in the index: small inserts are
folded into the last partially-filled shard, so adding a few at a time does not leave the index
fragmented or larger. The extra bytes each path writes while a call is in flight are reclaimed
shortly afterwards.

The other thing the row path buys is that the index **finishes merging** sooner after a very small
call. That is worth something only if you wait for the merge: you pass `await_completion=True`, or
you call `update_documents` / `upsert_documents` soon after inserting, which waits for that store's
pending inserts to merge first. In that case:

| index dim | row finishes merging sooner for a call of up to |
|---|---|
| 256 | 4 documents |
| 384 | 4 documents |
| 512 | 3 documents |
| 768 | 2 documents |
| 1024 | 2 documents |
| 1536 | no size — use bulk |

```python
# client next to the server: the default path, however few documents you add
store.add_documents(new_docs)

# client across a network, adding a few documents: avoid uploading a full block
store.add_documents(new_docs, use_row_insert=True)
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

## What each path uploads

This is the one place the difference is large, and it does not show up in the timings above
because those were measured with the client on the same machine as the server.

| index dim | bulk uploads per call, any size | row uploads per document |
|---|---|---|
| 256 | 7.9 MB | 61.5 KB |
| 384 | 11.8 MB | 61.5 KB |
| 512 | 15.8 MB | 61.5 KB |
| 768 | 23.7 MB | 61.5 KB |
| 1024 | 31.5 MB | 61.5 KB |
| 1536 | 47.3 MB | 61.5 KB |

Adding one document to a dim-1024 index uploads 31.5 MB on the bulk path and 61.5 KB on the row
path — a factor of 500. The timings earlier in this page were taken with the client on the same
machine as the server, where moving 31.5 MB is free and the difference does not appear.

Adding the transfer time to both columns gives the call size below which the row path returns
sooner. It barely depends on the dimension, because the bulk block, the row path's per-document
work and the upload all scale with it together — what decides it is the link:

| link to the server | row path returns sooner for calls of up to |
|---|---|
| 1 Gbps or same host | no size — use bulk |
| 100 Mbps | 1–2 documents |
| 50 Mbps | 2–3 documents |
| 10 Mbps | 9–12 documents |

These four rows are the measured timings plus transfer time at each speed, not four measurements;
time both paths if the choice is close for you.

## Server storage

Each call also makes the server write more than it keeps: the bulk path stores the uploaded block,
the row path about 2.1× that. Those are reclaimed a few minutes later and neither accumulates.

| index dim | written per call, bulk | written per call, row |
|---|---|---|
| 256 | 7.9 MB | 16.9 MB |
| 512 | 15.8 MB | 33.7 MB |
| 1024 | 31.5 MB | 67.2 MB |
| 1536 | 47.3 MB | 100.8 MB |

What stays is the same for both paths. After 45 small calls on an index of 8192 documents, at every
dimension, the index held three shards of 4096, 4096 and 402 documents — the small inserts were
folded into the last partly-filled shard rather than each leaving one of its own. A shard does have
a fixed cost (4.2 MB at dim 1024, plus about 4.1 KB per document), but it is paid once per 4096
documents, not once per call.

Measured on a single-node EnVector deployment, pyenvector 1.6.2, preset ip3, eval mode mms32,
FLAT index, with nothing else running against the server. Your numbers will differ with hardware,
index size and load, and the row path in particular degrades further when calls arrive faster than
the server merges them. The shape — bulk flat in the number of documents, row rising with it — is
a property of the two paths, so if the boundary matters to your deployment, time both at the size
you actually add.
