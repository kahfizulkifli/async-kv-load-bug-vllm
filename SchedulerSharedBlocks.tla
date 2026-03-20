---- MODULE SchedulerSharedBlocks ----
(*
  Extension of SchedulerAbstract that models shared KV-cache blocks.

  New variables
  ─────────────
  blocks[r]   ↔  the sequence of block IDs allocated to request r
                 (maps position index → block id)
                 Empty sequence = no blocks allocated yet.

  New safety properties
  ──────────────────────
  1. Original: backing[r] = credit[r] when r reaches ACTIVE.
  2. New: if block b is shared between r1 and r2, and r1 has b
     invalidated, r2's credit must NOT include tokens in block b
     unless r2 independently recomputed them.
  3. New: a block must not be freed while any request still
     holds it (ref count safety).

  Shared block scenario
  ──────────────────────
  r1 and r2 have the same prefix of length k (k blocks).
  Both go async. r1 gets InvalidBlockReport on block at index i < k.
  The shared block is marked invalid.
  r2 must also have its credit truncated to i * BlockSize.

  Implementation mapping
  ──────────────────────
  blocks[r]        ↔  kv_cache_manager.get_block_ids(req_id)
  SharedBlock(b)   ↔  block.ref_cnt > 1
  InvalidateShared ↔  _handle_invalid_blocks scanning both running
                       and skipped_waiting queues
*)

EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
  Requests,    \* finite set of request ids, e.g. {r1, r2}
  MaxN,        \* upper bound on prompt length in blocks
  NumBlocks    \* total number of block IDs available

BlockIds == 1..NumBlocks

\* Maximum number of blocks a request can hold = MaxN
\* (one block per N unit, since N is already in block units)

Phase == {
  "IDLE", "WAITING",
  "PROMISED", "PROMISED_ERR",
  "RECV_DONE",
  "CACHED",
  "ACTIVE", "DONE"
}

VARIABLES
  phase,    \* Requests -> Phase
  N,        \* Requests -> Nat   (prompt length in blocks)
  credit,   \* Requests -> Nat   (num_computed_tokens in blocks)
  backing,  \* Requests -> Nat   (tokens written to GPU cache, in blocks)
  blocks    \* Requests -> Seq(BlockIds)  (allocated block sequence)

vars == << phase, N, credit, backing, blocks >>

\* ─────────────────────────────────────────────────────────────────
\* Helpers
\* ─────────────────────────────────────────────────────────────────

\* Set of block IDs held by request r
BlocksOf(r) == {blocks[r][i] : i \in DOMAIN blocks[r]}

\* True if block b is shared between at least two requests
IsShared(b) ==
  Cardinality({r \in Requests : b \in BlocksOf(r)}) > 1

\* Set of requests that hold block b
Holders(b) == {r \in Requests : b \in BlocksOf(r)}

\* First index (1-based) in blocks[r] where block b appears
\* Returns 0 if not found
IndexOf(r, b) ==
  IF \E i \in DOMAIN blocks[r] : blocks[r][i] = b
  THEN CHOOSE i \in DOMAIN blocks[r] : blocks[r][i] = b
  ELSE 0

\* ─────────────────────────────────────────────────────────────────
Init ==
  /\ phase   = [r \in Requests |-> "IDLE"]
  /\ N       = [r \in Requests |-> 0]
  /\ credit  = [r \in Requests |-> 0]
  /\ backing = [r \in Requests |-> 0]
  /\ blocks  = [r \in Requests |-> << >>]

\* ─────────────────────────────────────────────────────────────────
\* Arrive: allocate a fresh block sequence for the request.
\* The first k blocks may overlap with another request's blocks
\* (shared prefix). We model this by allowing blocks[r] to be
\* ANY sequence of length n drawn from BlockIds, including
\* sequences that share a prefix with an existing request.
\* ─────────────────────────────────────────────────────────────────
Arrive(r, n, blks) ==
  /\ phase[r] = "IDLE"
  /\ n \in 1..MaxN
  /\ Len(blks) = n
  /\ \A i \in DOMAIN blks : blks[i] \in BlockIds
  \* NEW: no two positions within the same request share a block ID
  /\ \A i, j \in DOMAIN blks : i # j => blks[i] # blks[j]
  \* Blocks are shared with at most one other request (prefix sharing)
  /\ \A i \in DOMAIN blks :
       Cardinality(Holders(blks[i])) <= 1
  /\ phase'  = [phase  EXCEPT ![r] = "WAITING"]
  /\ N'      = [N      EXCEPT ![r] = n]
  /\ blocks' = [blocks EXCEPT ![r] = blks]
  /\ UNCHANGED << credit, backing >>

\* ─────────────────────────────────────────────────────────────────
\* PromiseAsync: same as before; blocks already allocated at Arrive.
\* ─────────────────────────────────────────────────────────────────
PromiseAsync(p) ==
  \* At least one WAITING request must exist with N >= p
  /\ \E r \in Requests : phase[r] = "WAITING" /\ N[r] >= p
  /\ p \in 1..MaxN
  \* Each WAITING request gets credit = min(p, N[r])
  \* This models the scheduler loop: each request gets its own
  \* connector call, but all happen in the same schedule() step.
  /\ phase' = [r \in Requests |->
       IF phase[r] = "WAITING"
       THEN "PROMISED"
       ELSE phase[r]]
  /\ credit' = [r \in Requests |->
       IF phase[r] = "WAITING"
       THEN IF p <= N[r] THEN p ELSE N[r]
       ELSE credit[r]]
  /\ UNCHANGED << N, backing, blocks >>

\* ─────────────────────────────────────────────────────────────────
\* InvalidBlockReport: a block is reported invalid.
\* 
\* KEY EXTENSION: if the invalid block is shared, ALL holders that
\* are still in PROMISED/PROMISED_ERR must have their credit
\* truncated.  This models _handle_invalid_blocks scanning both
\* running[] and skipped_waiting[].
\* ─────────────────────────────────────────────────────────────────
InvalidBlockReport(b) ==
  \* b must be held by at least one PROMISED/PROMISED_ERR request
  /\ \E r \in Requests :
       /\ phase[r] \in {"PROMISED", "PROMISED_ERR"}
       /\ b \in BlocksOf(r)
  \* For every holder in PROMISED/PROMISED_ERR, truncate credit
  \* to (index_of_b - 1), i.e. (IndexOf(r,b) - 1) blocks.
  /\ phase' = [r \in Requests |->
       IF phase[r] \in {"PROMISED", "PROMISED_ERR"} /\ b \in BlocksOf(r)
       THEN "PROMISED_ERR"
       ELSE phase[r]]
  /\ credit' = [r \in Requests |->
       IF phase[r] \in {"PROMISED", "PROMISED_ERR"} /\ b \in BlocksOf(r)
       THEN
         \* truncate to the index before b
         LET idx == IndexOf(r, b)
         IN  IF idx - 1 < credit[r] THEN idx - 1 ELSE credit[r]
       ELSE credit[r]]
  /\ UNCHANGED << N, backing, blocks >>

\* ─────────────────────────────────────────────────────────────────
\* FinishedRecving: unchanged.
\* ─────────────────────────────────────────────────────────────────
FinishedRecving(r) ==
  /\ phase[r] \in {"PROMISED", "PROMISED_ERR"}
  /\ phase'  = [phase EXCEPT ![r] = "RECV_DONE"]
  /\ UNCHANGED << N, credit, backing, blocks >>

\* ─────────────────────────────────────────────────────────────────
\* CacheBlocks: backs credit[r] tokens into GPU cache.
\* Shared blocks: if block i is shared, it is cached by the first
\* request to call CacheBlocks; the second request can reuse it.
\* No change to the block ownership — both still hold the block.
\* ─────────────────────────────────────────────────────────────────
CacheBlocks(r) ==
  /\ phase[r]  = "RECV_DONE"
  /\ credit[r] > 0
  /\ phase'   = [phase   EXCEPT ![r] = "CACHED"]
  /\ backing' = [backing EXCEPT ![r] = credit[r]]
  /\ UNCHANGED << N, credit, blocks >>

\* ─────────────────────────────────────────────────────────────────
\* CacheBlocksRetry: credit was 0; free blocks, re-check local cache.
\*
\* KEY CONSTRAINT: a shared block must NOT be freed.
\* new_credit is limited to blocks that are still present
\* (not freed because they were shared with another request).
\* ─────────────────────────────────────────────────────────────────
CacheBlocksRetry(r, new_credit) ==
  /\ phase[r]  = "RECV_DONE"
  /\ credit[r] = 0
  /\ new_credit \in 0..(N[r] - 1)
  \* Cannot claim credit beyond a shared block that was NOT freed
  \* (because freeing a shared block requires ref_cnt to drop to 0)
  /\ \A i \in 1..new_credit :
       \* block i is either not shared, or was already freed (not shared now)
       ~IsShared(blocks[r][i])
  /\ phase'   = [phase   EXCEPT ![r] = "CACHED"]
  /\ credit'  = [credit  EXCEPT ![r] = new_credit]
  /\ backing' = [backing EXCEPT ![r] = new_credit]
  /\ UNCHANGED << N, blocks >>

\* ─────────────────────────────────────────────────────────────────
\* ScheduleBacked / Execute / ScheduleDirect: unchanged from abstract.
\* ─────────────────────────────────────────────────────────────────
ScheduleDirect(r, c) ==
  /\ phase[r]   = "WAITING"
  /\ c \in 0..(N[r] - 1)
  /\ backing[r] >= c
  /\ phase'  = [phase  EXCEPT ![r] = "ACTIVE"]
  /\ credit' = [credit EXCEPT ![r] = c]
  /\ UNCHANGED << N, backing, blocks >>

ScheduleBacked(r) ==
  /\ phase[r]   = "CACHED"
  /\ credit[r]  < N[r]
  /\ backing[r] = credit[r]
  /\ phase'  = [phase EXCEPT ![r] = "ACTIVE"]
  /\ UNCHANGED << N, credit, backing, blocks >>

Execute(r) ==
  /\ phase[r] = "ACTIVE"
  /\ phase'   = [phase EXCEPT ![r] = "DONE"]
  /\ UNCHANGED << N, credit, backing, blocks >>

\* ─────────────────────────────────────────────────────────────────
Next ==
  \/ \E r \in Requests: \E n \in 1..MaxN:
        \E blks \in [1..n -> BlockIds] : Arrive(r, n, blks)
  \/ \E p \in 1..MaxN : PromiseAsync(p)
  \/ \E b \in BlockIds               : InvalidBlockReport(b)
  \/ \E r \in Requests               : FinishedRecving(r)
  \/ \E r \in Requests               : CacheBlocks(r)
  \/ \E r \in Requests, c \in 0..MaxN : CacheBlocksRetry(r, c)
  \/ \E r \in Requests               : ScheduleBacked(r)
  \/ \E r \in Requests               : Execute(r)
  \/ \E r \in Requests, c \in 0..MaxN : ScheduleDirect(r, c)

Spec == Init /\ [][Next]_vars

====