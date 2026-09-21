# Furnace

**Checkpointed distributed training with fault injection - training infra is judged on recovery, not the happy path.**

Furnace runs a real PyTorch training loop (tiny MLP, synthetic data, CPU-only) that gets *killed mid-run* by a fault injector, then restarts under a supervisor that proves resume-from-checkpoint: the loss curve continues instead of restarting from scratch.

## What it proves (with numbers)

### Single-rank fault drill (proven run)

```
$ python furnace.py --world-size 1 --max-steps 60 --ckpt-every 10 --fault-at-step 25

[rank0] step 10 loss 26.41489
[rank0] step 20 loss 27.83180
[rank0] FAULT INJECTION: hard kill at step 25 (no cleanup, no save)
=== attempt 2 ===
[rank0] RESUMED from step 20 (loss 27.8318)
[rank0] step 30 loss ...
[rank0] step 60 loss 15.17...
DONE attempts=2
```

The job died hard at step 25 (no cleanup, no save, like an OOM-kill or power loss). The last checkpoint was step 20. The supervisor restarted the job, rank 0 loaded the checkpoint (Adam moments included), and training continued from step 20 to the full 60 steps. Loss went 27.83 at the resume point down to 15.17 at completion - the curve continued, it did not restart.

### Two-rank fault drill (proven run)

```
$ python furnace.py --world-size 2 --max-steps 60 --ckpt-every 10 --fault-at-step 35

=== attempt 1 port 24845 (fault_left=35) ===
[rank0] step 30 loss 27.04852
[rank0] FAULT INJECTION: hard kill at step 35
[rank1] FAULT INJECTION: hard kill at step 35
supervisor: exitcodes [137, 137] - restarting
supervisor: fault consumed - restart runs clean
=== attempt 2 port 22179 (fault_left=None) ===
[rank0] RESUMED from step 30 (loss 27.0485)
[rank0] step 60 loss 21.48315
DONE attempts=2
```

Both ranks die at step 35 (`os._exit(137)` - a real job manager kills the whole allocation). The supervisor sees both exit codes, restarts with a fresh rendezvous port, rank 0 resumes from the step-30 checkpoint, and DDP training completes to step 60.

## Why these details matter

**Atomic checkpoints.** `atomic_save` writes to a `.tmp` file, fsyncs it, then `os.replace` (atomic rename). A kill at ANY instant leaves either the old checkpoint or the new one - never a torn file. Recovery code that assumes the worst (`load_latest` catches load errors) stays belt-and-suspenders.

**Adam memory math.** The model is not the memory bill - optimizer states are. Adam carries 2 extra fp32 copies of the parameters (`exp_avg` + `exp_avg_sq`), so the standing bill is 4x param size (1x params + 1x grads + 2x Adam). `shard_account` computes the FSDP-style per-rank bill: full replication costs `total` per rank, sharding costs `total / world_size` per rank at the cost of per-layer all-gather communication. For a 1M-param model with Adam at world_size=2: 0.016 GB replicated vs 0.008 GB sharded per rank. Scale that math to 70B params and the case for FSDP writes itself.

**Supervisor never hangs.** Three safeguards:
1. Fresh random rendezvous port per attempt - a killed attempt's lingering TIME_WAIT sockets can never wedge the next rendezvous.
2. `init_process_group(timeout=60s)` - a rank stuck waiting on a dead peer fails bounded instead of hanging forever.
3. Bounded join in the supervisor (60s with a fault pending, 900s clean) with kill-on-stall - the supervisor decides, it never hangs with its workers. After 10 failed attempts it gives up loudly rather than crash-looping forever.

## Windows-specific fix baked in

Windows CPU wheels of torch 2.4.x are built without libuv support but request it by default: TCPStore creation fails and every rank dies at init (which looks exactly like a hang). Furnace sets `USE_LIBUV=0` to force the legacy TCP store.

## Honest scope

- Both workers run on THIS machine (world_size=2 via torch.distributed gloo on localhost). The coordination logic is real distributed training; the hardware is one box. No cloud, no GPUs.
- The recovery machinery (atomic checkpoints, resume semantics, supervision) is identical at any scale.
- Not a multi-node GPU cluster run.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

# single-rank fault drill
.venv/Scripts/python furnace.py --world-size 1 --max-steps 60 --ckpt-every 10 --fault-at-step 25

# two-rank fault drill
.venv/Scripts/python furnace.py --world-size 2 --max-steps 60 --ckpt-every 10 --fault-at-step 35

# tests (offline, no distributed)
.venv/Scripts/python -m pytest tests/ -q
```

## Tests

6 offline unit tests (`tests/test_furnace.py`): atomic save roundtrip, no torn files, optimizer-state resume (Adam moments survive), shard-account math, deterministic dataloader, model forward shape. CI adds a single-rank fault-drill smoke run.
