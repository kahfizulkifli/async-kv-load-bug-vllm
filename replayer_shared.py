"""
Replayer for SchedulerSharedBlocks.tla

Two-part verification:
1. Scheduler invariant check: after InvalidBlockReport, does r2 retain
   credit > backing? (spec violation)
2. Output oracle: does the live engine_core (in the bugged state) produce
   different output for r2 than a clean full-recompute reference?
   - Run A: live engine_core as-is. r2 has num_computed_tokens=credit.
            Block 1 is in whatever state vLLM left it (zeros from
            _zero_block_ids, since the async load never completed).
            No oracle intervention on GPU memory.
   - Run B: fresh engine_core, same prompt, num_computed_tokens=0.
"""
from __future__ import annotations
import argparse, json, sys, gc
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock
import os

sys.path.append(os.environ["VLLM_PATH"])

from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_vllm_config,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)

BLOCK_SIZE = 16
EOS_TOKEN_ID = 50256
MODEL = "facebook/opt-125m"

# ─────────────────────────────────────────────────────────────────
# Request creation
# ─────────────────────────────────────────────────────────────────

_req_counter = 0

def _make_request(
    num_tokens: int,
    common_prefix_tokens: list[int] | None = None,
    block_size: int = BLOCK_SIZE,
) -> Request:
    global _req_counter
    _req_counter += 1

    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    prefix = common_prefix_tokens or []
    suffix_len = num_tokens - len(prefix)
    suffix = [_req_counter * 1000 + i for i in range(suffix_len)]
    prompt_token_ids = prefix + suffix

    return Request(
        request_id=f"id-{_req_counter}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


# ─────────────────────────────────────────────────────────────────
# EngineCore factory
# ─────────────────────────────────────────────────────────────────

def _make_engine_core():
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.engine.core import EngineCore
    from vllm.config.device import DeviceConfig
    from vllm.config import AttentionConfig

    vllm_config = create_vllm_config(
        model=MODEL,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=2048,
        kv_load_failure_policy="recompute",
    )
    vllm_config.device_config = DeviceConfig("cuda")
    vllm_config.attention_config = AttentionConfig(backend="FLASH_ATTN")
    vllm_config.kv_transfer_config.kv_connector = "OffloadingConnector"
    vllm_config.kv_transfer_config.kv_role = "kv_both"
    vllm_config.kv_transfer_config.kv_connector_extra_config["block_size"] = BLOCK_SIZE
    vllm_config.kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"] = 20_000_000
    vllm_config.scheduler_config.async_scheduling = False
    vllm_config.cache_config.gpu_memory_utilization = 0.1

    return EngineCore(
        vllm_config=vllm_config,
        executor_class=UniProcExecutor,
        log_stats=False,
    )


def make_connector() -> Mock:
    c = Mock()
    c.get_num_new_matched_tokens.return_value = (0, False)
    c.update_state_after_alloc.return_value   = None
    c.request_finished.return_value           = (False, None)
    c.take_events.return_value                = ()
    c.update_connector_output.return_value    = None
    c.build_connector_meta.return_value       = OffloadingConnectorMetadata({}, {})
    return c


def set_connector(connector, tokens, async_load):
    connector.get_num_new_matched_tokens.side_effect = None
    connector.get_num_new_matched_tokens.return_value = (tokens, async_load)


# ─────────────────────────────────────────────────────────────────
# Oracle helpers
# ─────────────────────────────────────────────────────────────────

def _collect_tokens(engine_core: EngineCore, req_id: str, max_steps: int = 10) -> list[int]:
    """
    Drive engine_core.step() until a token is generated for req_id,
    or max_steps is exhausted. Returns sampled token IDs.
    No GPU intervention — uses whatever is already in the KV cache.
    """
    engine_core.scheduler.connector.get_num_new_matched_tokens.return_value = (
        0, False
    )
    token_ids: list[int] = []
    for _ in range(max_steps):
        if not engine_core.scheduler.has_requests():
            break
        outputs, _ = engine_core.step()
        for _, eco in (outputs or {}).items():
            for out in eco.outputs or []:
                if out.request_id == req_id:
                    token_ids.extend(out.new_token_ids or [])
        if token_ids:
            break
        
    engine_core.shutdown()
    del engine_core
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    return token_ids


def _run_clean_reference(prompt_token_ids: list[int]) -> list[int]:
    """
    Run B: fresh engine_core, same prompt tokens, num_computed_tokens=0.
    Full recompute from scratch — no async load, no credit.
    Returns the first sampled token IDs.
    """
    engine_core = _make_engine_core()
    connector   = make_connector()
    engine_core.scheduler.connector = connector

    from vllm import SamplingParams
    init_none_hash(sha256)
    sp = SamplingParams(max_tokens=16, temperature=0.0)
    sp.update_from_generation_config({}, EOS_TOKEN_ID)
    req = Request(
        request_id="clean-ref",
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=sp,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    engine_core.scheduler.add_request(req)

    # Sync connector — no async load, no credit.
    connector.get_num_new_matched_tokens.return_value = (0, False)

    try:
        return _collect_tokens(engine_core, req.request_id)
    finally:
        try:
            engine_core.shutdown()
        except Exception:
            pass
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────
# Graph loading
# ─────────────────────────────────────────────────────────────────

@dataclass
class TLCState:
    fp: int
    val: dict[str, Any]
    initial: bool = False

@dataclass
class TLCEdge:
    from_fp: int
    to_fp:   int
    act:     str
    params:  dict[str, Any]

def load_graph(states_path, edges_path):
    with open(states_path) as f:
        states = {s["fp"]: TLCState(s["fp"], s["val"], s.get("initial", False))
                  for s in json.load(f)["states"]}
    with open(edges_path) as f:
        adj = defaultdict(list)
        for e in json.load(f)["edges"]:
            adj[e["from"]].append(
                TLCEdge(e["from"], e["to"], e["act"], e.get("params", {}))
            )
    return states, adj

def is_true_initial(state: TLCState) -> bool:
    val = state.val
    return (
        all(p == "IDLE" for p in val.get("phase",   {}).values())
        and all(int(c) == 0 for c in val.get("credit",  {}).values())
        and all(int(n) == 0 for n in val.get("N",       {}).values())
        and all(int(b) == 0 for b in val.get("backing", {}).values())
    )

def is_corrupt_tlc(state: TLCState) -> bool:
    phase   = state.val.get("phase",   {})
    credit  = state.val.get("credit",  {})
    backing = state.val.get("backing", {})
    return any(
        phase.get(r) == "ACTIVE"
        and int(credit.get(r, 0)) > int(backing.get(r, 0))
        for r in phase
    )

def path_has_meaningful_cache_action(path, states):
    for edge in path:
        if edge.act not in {"CacheBlocks", "CacheBlocksRetry"}:
            continue
        n_vals = states[edge.from_fp].val.get("N", {})
        if any(int(n) >= 2 for n in n_vals.values()):
            return True
    return False

def precompute_can_reach_cache(states: dict, adj: dict) -> set[int]:
    CACHE_ACTIONS = {"CacheBlocks", "CacheBlocksRetry",
                     "BuggyCacheBlocks", "CacheAndSchedule"}
    initial = {fp for fp, s in states.items() if is_true_initial(s)}
    print(f"  [diagnose] true initial states: {len(initial)}")

    forward_reachable: set[int] = set(initial)
    frontier = list(initial)
    while frontier:
        nxt = []
        for fp in frontier:
            for e in adj.get(fp, []):
                if e.to_fp not in forward_reachable:
                    forward_reachable.add(e.to_fp)
                    nxt.append(e.to_fp)
        frontier = nxt
    print(f"  [diagnose] forward reachable: {len(forward_reachable)}")

    has_cache_edge = {
        fp for fp in forward_reachable
        if any(e.act in CACHE_ACTIONS for e in adj.get(fp, []))
    }
    print(f"  [diagnose] states with cache edge: {len(has_cache_edge)}")
    if not has_cache_edge:
        return forward_reachable

    rev_adj: dict[int, list[int]] = defaultdict(list)
    for fp in forward_reachable:
        for e in adj.get(fp, []):
            if e.to_fp in forward_reachable:
                rev_adj[e.to_fp].append(fp)

    can_reach: set[int] = set(has_cache_edge)
    frontier = list(has_cache_edge)
    while frontier:
        nxt = []
        for fp in frontier:
            for pred in rev_adj[fp]:
                if pred not in can_reach:
                    can_reach.add(pred)
                    nxt.append(pred)
        frontier = nxt
    print(f"  [diagnose] can_reach_cache: {len(can_reach)}")
    return can_reach

def find_all_paths(states, adj, max_paths=50, max_depth=20,
                   target="corrupt") -> list[list[TLCEdge]]:
    paths: list[list[TLCEdge]] = []
    seen: set[tuple] = set()
    can_reach_cache = precompute_can_reach_cache(states, adj)

    def path_sig(path):
        return tuple((e.act, tuple(sorted(e.params.items()))) for e in path)

    def dfs(fp, path, visited):
        if len(paths) >= max_paths:
            return
        s = states.get(fp)
        if s is None or fp not in can_reach_cache:
            return
        if target == "corrupt" and is_corrupt_tlc(s):
            sig = path_sig(path)
            if sig not in seen and path_has_meaningful_cache_action(path, states):
                seen.add(sig)
                paths.append(list(path))
            return
        if fp in visited:
            return
        sig = path_sig(path)
        if sig in seen:
            return
        if target == "any" and path_has_meaningful_cache_action(path, states):
            seen.add(sig)
            paths.append(list(path))
        visited.add(fp)
        for e in adj.get(fp, []):
            path.append(e)
            dfs(e.to_fp, path, visited)
            path.pop()
        visited.discard(fp)

    for fp, s in states.items():
        if is_true_initial(s):
            dfs(fp, [], set())
    return paths


# ─────────────────────────────────────────────────────────────────
# Replay state
# ─────────────────────────────────────────────────────────────────
from vllm.v1.engine.core import EngineCore
@dataclass
class ReplayState:
    engine_core:       EngineCore                           # vllm EngineCore
    sched:             Scheduler
    connector:         Mock
    last_async_tokens: int                   = field(default=0)
    req_map:           dict[str, Request]    = field(default_factory=dict)
    tlc_to_phys:       dict[int, int]        = field(default_factory=dict)
    tlc_blocks:        dict[str, list[int]]  = field(default_factory=dict)
    tlc_blocks_to_req: dict[int, str]        = field(default_factory=dict)
    blocks_in_flight:  set[int]              = field(default_factory=set)
    last_output:       SchedulerOutput|None  = field(default=None)
    shared_prefix:     list[int]             = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# Edge handlers
# ─────────────────────────────────────────────────────────────────

def _handle_arrive(rs: ReplayState, params: dict, states: dict,
                   to_fp: int) -> None:
    r        = params["r"]
    n        = int(params["n"])
    blks_str = params.get("blks", [])
    if isinstance(blks_str, str) and blks_str.startswith("<<"):
        blks_str = blks_str.replace("<<", "").replace(">>", "").split(",")
    blks = [int(b) for b in blks_str]
    num_tokens = n * BLOCK_SIZE

    shared_len    = 0
    prefix_tokens: list[int] = []
    for i, tlc_bid in enumerate(blks):
        if tlc_bid in rs.tlc_blocks_to_req:
            if i == shared_len:
                shared_len += 1
                existing_r   = rs.tlc_blocks_to_req[tlc_bid]
                existing_req = rs.req_map[existing_r]
                prefix_tokens = list(
                    existing_req.prompt_token_ids[:shared_len * BLOCK_SIZE]
                )
            else:
                break

    req = _make_request(num_tokens=num_tokens,
                        common_prefix_tokens=prefix_tokens)
    rs.sched.add_request(req)
    rs.req_map[r] = req

    rs.tlc_blocks[r] = blks
    for tlc_bid in blks:
        rs.tlc_blocks_to_req[tlc_bid] = r


def _handle_promise_async(rs: ReplayState, params: dict) -> None:
    p = int(params["p"])
    rs.last_async_tokens = p * BLOCK_SIZE
    set_connector(rs.sched.connector, p * BLOCK_SIZE, async_load=True)
    sched_out      = rs.sched.schedule()
    rs.last_output = sched_out
    # Sync model runner: registers new requests without running the model.
    # total_num_scheduled_tokens must be 0 (requests went to skipped_waiting).
    _sync_model_runner(rs, sched_out)

    # Build tlc_to_phys now that blocks are allocated.
    for r, req in rs.req_map.items():
        if req.request_id not in rs.sched.requests:
            continue
        try:
            (phys_blocks,) = rs.sched.kv_cache_manager.get_block_ids(
                req.request_id
            )
        except Exception:
            continue
        for i, tlc_bid in enumerate(rs.tlc_blocks.get(r, [])):
            if tlc_bid not in rs.tlc_to_phys and i < len(phys_blocks):
                rs.tlc_to_phys[tlc_bid] = phys_blocks[i]

    # Record which physical blocks are currently being loaded
    # (allocated but not yet written by a successful async load).
    rs.blocks_in_flight = set(rs.tlc_to_phys.values())


def _handle_invalid_block_report(rs: ReplayState, params: dict) -> None:
    tlc_b = int(params["b"])
    if rs.last_output is None:
        return

    phys_b = rs.tlc_to_phys.get(tlc_b)
    if phys_b is None:
        print(f"  [warn] InvalidBlockReport: TLC block {tlc_b} not in "
              f"tlc_to_phys {rs.tlc_to_phys} — skipping")
        return

    set_connector(rs.sched.connector, rs.last_async_tokens, async_load=True)
    sched_out      = rs.sched.schedule()
    rs.last_output = sched_out
    _sync_model_runner(rs, sched_out)

    mro = create_model_runner_output(
        reqs=list(rs.sched.running),
        finished_recving=None,
        invalid_block_ids={phys_b},
        use_eos=False,
    )
    rs.sched.update_from_output(sched_out, mro)


def _handle_finished_recving(rs: ReplayState, params: dict) -> None:
    r   = params["r"]
    req = rs.req_map.get(r)
    if req is None or rs.last_output is None:
        return

    set_connector(rs.sched.connector, rs.last_async_tokens, async_load=True)
    sched_out      = rs.sched.schedule()
    rs.last_output = sched_out
    _sync_model_runner(rs, sched_out)

    mro = create_model_runner_output(
        reqs=list(rs.sched.running),
        finished_recving={req.request_id},
        invalid_block_ids=None,
        use_eos=False,
    )
    rs.sched.update_from_output(sched_out, mro)
    
def _handle_cache_blocks(rs: ReplayState, params: dict,
                         states: dict, to_fp: int) -> tuple[bool, str, int, int]:
    r   = params["r"]
    req = rs.req_map.get(r)
    if req is None:
        return False, "", 0, 0

    tlc_backing_blocks = int(states[to_fp].val["backing"][r])
    safe_up_to_tokens  = tlc_backing_blocks * BLOCK_SIZE

    # From this point on, use engine_core.step() so the model runner
    # stays in sync. The connector is already set to (0, False).
    set_connector(rs.sched.connector, 0, False)
    
    # Drive one step to let r1 be scheduled and executed (CacheBlocksRetry).
    # The model runner will receive the NewRequestData for r1 here.
    step_outputs, model_executed = rs.engine_core.step()
    if model_executed:
        # Collect any tokens from this step (r1's prefill result).
        pass  # we don't need r1's tokens, just need to advance state

    credit_tokens  = rs.sched.requests[req.request_id].num_computed_tokens
    backing_tokens = safe_up_to_tokens
    
    if req.request_id not in rs.sched.requests:
        print(f"  ○ KNOWN SPEC GAP (freed)  req={req.request_id} "
              f"credit={credit_tokens} backing={backing_tokens}")
        return False, req.request_id, credit_tokens, backing_tokens

    tlc_credit = int(states[to_fp].val["credit"][r]) * BLOCK_SIZE
    suspect    = (credit_tokens > backing_tokens) or (credit_tokens != tlc_credit)

    if not suspect:
        return False, req.request_id, credit_tokens, backing_tokens

    # ── Check sibling requests for the shared-block bug ───────────────────
    for other_r, other_req in rs.req_map.items():
        if other_r == r:
            continue
        if other_req.request_id not in rs.sched.requests:
            continue

        other_credit  = rs.sched.requests[other_req.request_id].num_computed_tokens
        other_backing = int(states[to_fp].val["backing"].get(other_r, 0)) * BLOCK_SIZE

        if other_credit <= other_backing:
            continue

        # Invariant violation confirmed for sibling.
        print(f"  ! INVARIANT VIOLATION: {other_r} "
              f"credit={other_credit} > backing={other_backing}")

        # r2 is still in WAITING_FOR_REMOTE_KVS — deliver finished_recving
        # so the next step() can promote and schedule it.
        set_connector(rs.sched.connector, rs.last_async_tokens, async_load=True)
        sched_out = rs.sched.schedule()
        _sync_model_runner(rs, sched_out)

        mro = create_model_runner_output(
            reqs=list(rs.sched.running),   # r1 may still be running
            finished_recving={other_req.request_id},
            invalid_block_ids=None,
            use_eos=False,
        )
        rs.sched.update_from_output(sched_out, mro)
        # r2 is now in finished_recving_kv_req_ids.
        # Next step() will promote it to WAITING and then schedule it.

        # ── Run A: live engine_core, bugged state ─────────────────────────
        # other_req has num_computed_tokens=other_credit.
        # The shared block is in the state vLLM left it after allocation
        # and failed async load — zeroed by _zero_block_ids, never written
        # by a successful load. No oracle intervention on GPU memory.
        corrupt_toks = _collect_tokens(rs.engine_core, other_req.request_id)
        
        # Shut down live engine before starting clean reference
        # to avoid two EngineCore instances competing for GPU memory.
        rs.engine_core.shutdown()
        del rs.engine_core
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass


        # ── Run B: fresh engine_core, full recompute ──────────────────────
        clean_toks = _run_clean_reference(list(other_req.prompt_token_ids))

        print(f"    Run A (bugged):  {corrupt_toks[:6]}")
        print(f"    Run B (clean):   {clean_toks[:6]}")

        if corrupt_toks != clean_toks:
            print(f"  ✗ BUG CONFIRMED: outputs differ")
            return True, other_req.request_id, other_credit, other_backing
        else:
            # Block was repaired by r1's recompute before r2 ran.
            # The invariant was violated but the specific execution
            # self-corrected. This is still a real bug — a different
            # path ordering would expose wrong output.
            print(f"  ~ INVARIANT VIOLATED but outputs match "
                  f"(r1 repaired block before r2 ran)")
            return False, other_req.request_id, other_credit, other_backing

    # ── Primary request check ─────────────────────────────────────────────
    base_explanation = (
        f"credit={credit_tokens} > backing={backing_tokens}"
        if credit_tokens > backing_tokens
        else f"credit={credit_tokens} != tlc_credit={tlc_credit}"
    )
    corrupt_toks = _collect_tokens(rs.engine_core, req.request_id)
    clean_toks   = _run_clean_reference(list(req.prompt_token_ids))

    print(f"    Run A (bugged):  {corrupt_toks[:6]}")
    print(f"    Run B (clean):   {clean_toks[:6]}")

    if corrupt_toks != clean_toks:
        print(f"  ✗ BUG CONFIRMED: {base_explanation}")
        return True, req.request_id, credit_tokens, backing_tokens
    else:
        print(f"  ~ {base_explanation} but outputs match")
        return False, req.request_id, credit_tokens, backing_tokens

def _sync_model_runner(rs: ReplayState, sched_out: SchedulerOutput) -> None:
    """
    Sync the model runner's internal state with a scheduler output.
    Only valid when total_num_scheduled_tokens == 0 (no model execution).
    Registers new requests in model_runner.requests without running the model.
    """
    if sched_out.total_num_scheduled_tokens > 0:
        # r1 is already running and was included in this schedule output.
        # All requests are already registered in the model runner from
        # the PromiseAsync sync. Nothing new to register — skip.
        assert not sched_out.scheduled_new_reqs, (
            f"Unexpected new requests in non-empty schedule output: "
            f"{[r.req_id for r in sched_out.scheduled_new_reqs]}"
        )
        return
    model_runner = rs.engine_core.model_executor.driver_worker.model_runner
    model_runner._update_states(sched_out)

# ─────────────────────────────────────────────────────────────────
# Path replay
# ─────────────────────────────────────────────────────────────────

def replay_path(path, states, path_idx) -> bool:
    engine_core = _make_engine_core()
    connector   = make_connector()
    engine_core.scheduler.connector = connector

    rs = ReplayState(
        engine_core=engine_core,
        sched=engine_core.scheduler,
        connector=connector,
    )

    print(f"\nPath {path_idx}: {' → '.join(e.act for e in path)}")

    # try:
    for edge in path:
        act, params = edge.act, edge.params
        print(f"  {act} {params}")
        if act == "Arrive":
            _handle_arrive(rs, params, states, edge.to_fp)
        elif act == "PromiseAsync":
            _handle_promise_async(rs, params)
        elif act == "InvalidBlockReport":
            _handle_invalid_block_report(rs, params)
        elif act == "FinishedRecving":
            _handle_finished_recving(rs, params)
        elif act in ("CacheBlocks", "CacheBlocksRetry"):
            corrupt, req_id, credit, backing = _handle_cache_blocks(
                rs, params, states, edge.to_fp)
            if corrupt:
                return True
            print(f"    OK: credit={credit} backing={backing}")
        elif act in ("ScheduleBacked", "Execute", "ScheduleDirect"):
            pass
        else:
            print(f"  [unknown action] {act}")

    try:
        if rs.engine_core is not None:  # may have been shut down in _handle_cache_blocks
            rs.engine_core.shutdown()
    except Exception:
        pass

    print("  [path complete — no corruption]")
    return False


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states",    required=True)
    ap.add_argument("--edges",     required=True)
    ap.add_argument("--max-paths", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--target",    default="corrupt",
                    choices=["corrupt", "any"])
    args = ap.parse_args()

    states, adj = load_graph(args.states, args.edges)
    print(f"Loaded {len(states)} states, "
          f"{sum(len(v) for v in adj.values())} edges")

    print(f"Actions: {sorted({e.act for es in adj.values() for e in es})}")
    initial_fps = [fp for fp, s in states.items() if s.initial]
    print(f"Initial states: {len(initial_fps)}")
    can_reach = precompute_can_reach_cache(states, adj)
    print(f"Initial states in can_reach: "
          f"{sum(1 for fp in initial_fps if fp in can_reach)}")

    paths = find_all_paths(states, adj, args.max_paths, args.max_depth,
                           args.target)
    print(f"Found {len(paths)} paths (target={args.target})")
    if not paths:
        print("No paths found.")
        return

    new_bugs = 0
    paths = paths[6:]
    for i, path in enumerate(paths, 1):
        if replay_path(path, states, i):
            new_bugs += 1

    print(f"\n{'='*55}")
    print(f"Bugs confirmed by output divergence: {new_bugs}/{len(paths)}")
    if new_bugs == 0:
        print("✓ No output divergence. Invariant violations found but "
              "outputs self-corrected in these specific paths.")
    else:
        print("✗ Bugs confirmed.")


if __name__ == "__main__":
    main()