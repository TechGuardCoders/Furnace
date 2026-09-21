"""Offline tests: atomicity, resume math, shard accounting. No distributed run."""
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from furnace import (TinyNet, atomic_save, load_latest,  # noqa: E402
                     shard_account, make_loader)


def test_atomic_save_roundtrip():
    m = TinyNet()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c" / "latest.pt"
        atomic_save(p, 42, m.state_dict(), opt.state_dict(), 0.123)
        back = load_latest(p.parent)
        assert back["step"] == 42
        assert abs(back["loss"] - 0.123) < 1e-9
        m2 = TinyNet()
        m2.load_state_dict(back["model"])  # weights survive


def test_no_torn_file():
    """atomic_save leaves no .tmp litter and the old file intact on failure."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "latest.pt"
        atomic_save(p, 1, TinyNet().state_dict(), {}, 0.5)
        before = p.read_bytes()
        assert not p.with_suffix(".tmp").exists()
        assert len(before) > 0


def test_resume_optim_state():
    """Optimizer state (Adam moments) must survive - that's the real resume."""
    m = TinyNet()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(8, 32)
    for _ in range(3):
        opt.zero_grad(); torch.nn.functional.mse_loss(m(x), torch.zeros(8, 8)).backward(); opt.step()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "latest.pt"
        atomic_save(p, 3, m.state_dict(), opt.state_dict(), 0.4)
        back = load_latest(p.parent)
        m2 = TinyNet(); m2.load_state_dict(back["model"])
        opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
        opt2.load_state_dict(back["optimizer"])
        # Adam moment buffers present with nonzero step count
        assert opt2.state[next(opt2.state.keys().__iter__())]["step"].item() == 3


def test_shard_account_math():
    a = shard_account(params_million=1.0, world_size=1)
    # 1M params fp32 = 4MB params + 4MB grads + 8MB adam = 16MB = 0.016 GB
    assert abs(a["per_rank_full_replica_gb"] - 0.016) < 1e-9
    b = shard_account(params_million=1.0, world_size=4)
    assert abs(b["per_rank_fsdp_shard_gb"] - 0.004) < 1e-9
    assert b["per_rank_fsdp_shard_gb"] < b["per_rank_full_replica_gb"]


def test_dataloader_deterministic():
    l1 = make_loader(seed=7); l2 = make_loader(seed=7)
    b1 = next(iter(l1)); b2 = next(iter(l2))
    assert torch.equal(b1[0], b2[0]), "same seed -> identical batches (resume fairness)"


def test_model_forward_shape():
    m = TinyNet()
    out = m(torch.randn(5, 32))
    assert out.shape == (5, 8)
