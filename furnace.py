"""
Furnace - checkpointed distributed training with fault injection.

The thesis: training infra is judged on recovery, not the happy path.

What this project IS (honest scope):
- A real PyTorch training loop (tiny model, synthetic data) that checkpoints
  atomically (tmp + fsync + rename) every N steps
- A fault injector that KILLS the process (SIGKILL-class: os._exit) mid-run
- A supervisor that restarts the job and proves resume-from-checkpoint:
  loss continues the curve instead of restarting from scratch
- Shard accounting: optimizer states are the real memory bill (2x params for
  Adam); FSDP-style sharding math computed and reported per rank
- Both workers run on THIS machine (world_size=2 via torch.distributed on
  localhost/gloo) - the coordination logic is real distributed training,
  the hardware is one box. No cloud, no cluster mutation.

What this is NOT: a multi-node GPU cluster run. The DGX Sparks run the
production serving stack and are read-only. This trains on CPU with a tiny
model - the recovery machinery is identical at any scale.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------- fleet
#
# Fleet nodes: named serving nodes in the cluster. Furnace trains locally, but
# checkpoint TARGETS are per node - each node gets its own checkpoint directory
# so a per-node resume never mixes state. Default (no --node) keeps the original
# single-directory behavior unchanged. Read-only w.r.t. the serving stack.

FLEET_NODES = {
    "glm-head": "http://192.168.0.182:8000",
    "tensorfold": "http://192.168.0.157:8421",
    "bonsai-1": "http://192.168.0.10:8422",
    "bonsai-2": "http://192.168.0.144:8422",
    "sushi": "http://192.168.0.11:8450",
}


def checkpoint_dir_for(node: str | None, base_dir: str = "checkpoints") -> Path:
    """Per-node checkpoint target: <base>/<node>/ when a fleet node is named,
    otherwise the plain base dir (default single-node behavior)."""
    if node is None:
        return Path(base_dir)
    if node not in FLEET_NODES:
        raise SystemExit(f"unknown node {node!r}; known: {', '.join(sorted(FLEET_NODES))}")
    return Path(base_dir) / node


# ---------------------------------------------------------------- model


class TinyNet(nn.Module):
    """3-layer MLP on synthetic regression. Small enough for CPU, same
    structure (params/optimizer/shards) as a real training job."""

    def __init__(self, dim_in: int = 32, hidden: int = 128, dim_out: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(seed: int = 7, batch: int = 64, n: int = 4096):
    """Deterministic loader: seeded data AND a seeded shuffle generator, so a
    resumed run sees the exact same batch sequence as the original."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 32, generator=g)
    w = torch.randn(32, 8, generator=g)
    y = x @ w + 0.1 * torch.randn(n, 8, generator=g)
    shuffle_g = torch.Generator().manual_seed(seed)  # separate generator for shuffle
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True,
                      generator=shuffle_g)


# ------------------------------------------------- shard accounting (the WHY)


def shard_account(params_million: float, world_size: int, optimizer: str = "adam") -> dict:
    """FSDP-style memory math. The lesson: the MODEL is not the bill -
    optimizer states are (Adam: 2 extra copies) plus activations.

    Full replication (DDP): every rank holds everything.
    FSDP sharding: params+grads+optimizer split across ranks (~1/N each),
    materialized per-layer during forward/backward (all-gather).
    """
    param_gb = params_million * 4 / 1000          # fp32 params
    grad_gb = param_gb                            # gradients, same size
    opt_mult = {"adam": 2, "sgd": 0, "adamw": 2}[optimizer.lower()]
    opt_gb = param_gb * opt_mult                  # exp_avg + exp_avg_sq
    total_gb = param_gb + grad_gb + opt_gb

    return {
        "params_million": params_million,
        "world_size": world_size,
        "per_rank_full_replica_gb": round(total_gb, 3),
        "per_rank_fsdp_shard_gb": round(total_gb / world_size, 3),
        "note": f"Adam adds {opt_mult}x param size in optimizer states; "
                f"FSDP divides the standing bill by world_size at the cost of "
                f"all-gather communication per layer",
    }


# ------------------------------------------------------------ checkpointing


def atomic_save(path: Path, step: int, model_state: dict, opt_state: dict, loss: float) -> None:
    """Crash-safe save: write tmp, fsync, atomic rename. A kill at ANY instant
    leaves either the old checkpoint or the new one - never a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {"step": step, "model": model_state, "optimizer": opt_state, "loss": loss}
    torch.save(payload, tmp)
    # fsync the file contents (open write-handle; a read-handle fd cannot
    # fsync on Windows), then rename atomically
    fd = os.open(tmp, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)  # atomic on POSIX and Windows


def load_latest(ckpt_dir: Path) -> dict | None:
    ckpt = ckpt_dir / "latest.pt"
    if not ckpt.exists():
        return None
    try:
        return torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception:
        # torn checkpoint (should not happen with atomic_save, but recovery
        # code assumes the worst): fall back to nothing rather than crash-loop
        return None


# ---------------------------------------------------------------- worker


def train(rank: int, world_size: int, args, fault: dict) -> dict:
    """One training worker. `fault` = {"at_step": int|None} - when the global
    step hits it, the process dies hard (os._exit, no cleanup, no save)."""
    ddp = world_size > 1
    if ddp:
        # set (not setdefault): each supervisor attempt uses a FRESH port so a
        # killed prior attempt's lingering sockets/TIME_WAIT never wedge the
        # next rendezvous (the world_size=2 hang)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(args.port)
        # Windows CPU wheels of torch 2.4.x claim libuv but are built without
        # it: TCPStore creation fails and every rank dies at init (the
        # world_size=2 hang). Force the legacy TCP store.
        os.environ["USE_LIBUV"] = "0"
        dist.init_process_group("gloo", rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=60))
        torch.manual_seed(42 + rank)
    else:
        torch.manual_seed(42)

    model = TinyNet()
    if ddp:
        model = nn.parallel.DistributedDataParallel(model)
    raw_model = model.module if ddp else model
    opt = optim.Adam(raw_model.parameters(), lr=1e-3)
    lossfn = nn.MSELoss()
    loader = make_loader()

    start_step, start_loss = 0, None
    if rank == 0:
        prev = load_latest(Path(args.ckpt_dir))
        if prev:
            raw_model.load_state_dict(prev["model"])
            opt.load_state_dict(prev["optimizer"])
            start_step = prev["step"]
            start_loss = prev["loss"]
            print(f"[rank{rank}] RESUMED from step {start_step} (loss {start_loss:.4f})",
                  flush=True)
    if ddp:
        # broadcast resume state so ranks agree
        state = torch.tensor([float(start_step)], dtype=torch.float64)
        dist.broadcast(state, src=0)
        start_step = int(state.item())

    model.train()
    step = start_step
    history = []
    killed = False
    t0 = time.time()

    epochs = 2 if not args.quick else 1
    while step < args.max_steps:
        for xb, yb in loader:
            if step >= args.max_steps:
                break
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()
            step += 1

            if ddp:
                avg = loss.detach().clone()
                dist.all_reduce(avg, op=dist.ReduceOp.SUM)
                avg /= world_size
                loss_val = avg.item()
            else:
                loss_val = loss.item()

            if step % 10 == 0 or step == 1:
                history.append({"step": step, "loss": round(loss_val, 5)})
                if rank == 0:
                    print(f"[rank{rank}] step {step} loss {loss_val:.5f}", flush=True)

            if fault.get("at_step") == step and not killed:
                killed = True
                print(f"[rank{rank}] FAULT INJECTION: hard kill at step {step} "
                      f"(no cleanup, no save - like OOM-kill/power loss)", flush=True)
                sys.stdout.flush()
                time.sleep(0.05)
                os._exit(137)     # SIGKILL-class death, EVERY rank dies
                                  # (real job managers kill the whole allocation)

            if step % args.ckpt_every == 0 and rank == 0:
                atomic_save(Path(args.ckpt_dir) / "latest.pt", step,
                            raw_model.state_dict(), opt.state_dict(), loss_val)

    result = {"rank": rank, "final_step": step, "history": history,
              "wall_s": round(time.time() - t0, 1)}
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    return result


# ---------------------------------------------------------------- supervisor


def run_job(args, fault: dict) -> dict:
    """Supervisor: launch workers, restart on death, until job completes.
    Mirrors what slurm/Ray do: your job dies, the SUPERVISOR decides."""
    world = args.world_size
    attempts = 0
    fault_left = dict(fault)

    while True:
        attempts += 1
        # fresh RANDOM rendezvous port every attempt: a killed prior attempt's
        # lingering sockets/TIME_WAIT never wedge the next rendezvous, and a
        # bounded range can never overflow past 65535 (the world_size=2 hang)
        args.port = random.randint(20000, 60000)
        print(f"=== attempt {attempts} port {args.port} "
              f"(fault_left={fault_left.get('at_step')}) ===", flush=True)
        import torch.multiprocessing as mp
        procs = []
        for r in range(world):
            p = mp.Process(target=_worker_entry, args=(r, world, args, fault_left),
                           daemon=False)
            p.start()
            procs.append(p)

        # bounded join: a worker that hangs (e.g. collective waiting on a dead
        # peer) gets killed at the timeout - the supervisor must not hang with it
        join_timeout = 60 if fault_left.get("at_step") else 900
        deadline = time.time() + join_timeout
        while any(p.is_alive() for p in procs) and time.time() < deadline:
            time.sleep(0.5)
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(5)
        codes = [p.exitcode for p in procs]

        if all(c == 0 for c in codes):
            return {"attempts": attempts, "completed": True}
        print(f"supervisor: exitcodes {codes} - restarting", flush=True)
        # the fault fired (os._exit(137) on every rank): clear it, or the
        # resumed run just gets killed again at the same step - forever
        if fault_left.get("at_step") is not None and any(c == 137 for c in codes):
            fault_left["at_step"] = None
            print("supervisor: fault consumed - restart runs clean", flush=True)
        elif attempts >= 10:
            raise RuntimeError(
                f"supervisor: giving up after {attempts} attempts "
                f"(exitcodes {codes}) - not crash-looping forever")


def _worker_entry(rank, world, args, fault):
    result = train(rank, world, args, fault)
    if rank == 0:
        out = Path(args.ckpt_dir) / "result.json"
        out.write_text(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--port", type=int, default=29511)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fault-at-step", type=int, default=None,
                    help="hard-kill the job when global step hits this")
    ap.add_argument("--node", default=None,
                    help=f"fleet node to checkpoint for (one of: "
                         f"{', '.join(sorted(FLEET_NODES))}); checkpoints go to "
                         f"<ckpt-dir>/<node>/")
    args = ap.parse_args()

    # per-node checkpoint target: default (no --node) is the plain ckpt-dir
    args.ckpt_dir = str(checkpoint_dir_for(args.node, args.ckpt_dir))

    fault = {"at_step": args.fault_at_step}
    print(f"Furnace: world_size={args.world_size} max_steps={args.max_steps} "
          f"ckpt_every={args.ckpt_every} fault_at_step={args.fault_at_step} "
          f"node={args.node or 'default'} ckpt_dir={args.ckpt_dir}", flush=True)

    summary = run_job(args, fault)

    ckpt_dir = Path(args.ckpt_dir)
    (ckpt_dir / "summary.json").write_text(json.dumps({
        "attempts": summary["attempts"], "fault_injected": fault["at_step"],
        "world_size": args.world_size, "max_steps": args.max_steps,
        "node": args.node,
        "shard_account_1m_adam": shard_account(1.0, args.world_size),
    }, indent=2))
    print(f"DONE attempts={summary['attempts']}", flush=True)


if __name__ == "__main__":
    main()
