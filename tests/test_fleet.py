"""Fleet tests: per-node checkpoint targets. No distributed run, no network."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from furnace import FLEET_NODES, checkpoint_dir_for  # noqa: E402


def test_default_unchanged():
    """No node -> plain checkpoint dir (original single-node behavior)."""
    assert checkpoint_dir_for(None) == Path("checkpoints")
    assert checkpoint_dir_for(None, "ck") == Path("ck")


def test_per_node_targets():
    for name in FLEET_NODES:
        assert checkpoint_dir_for(name, "ck") == Path("ck") / name


def test_nodes_are_isolated():
    """Two nodes never share a checkpoint directory."""
    a = checkpoint_dir_for("glm-head", "ck")
    b = checkpoint_dir_for("sushi", "ck")
    assert a != b


def test_unknown_node_exits():
    with pytest.raises(SystemExit):
        checkpoint_dir_for("nope")


def test_node_roundtrip_with_atomic_save():
    """A per-node target is a real checkpoint dir: save + load works."""
    import tempfile

    import torch

    from furnace import TinyNet, atomic_save, load_latest

    with tempfile.TemporaryDirectory() as d:
        target = checkpoint_dir_for("tensorfold", d)
        atomic_save(target / "latest.pt", 7, TinyNet().state_dict(), {}, 0.25)
        back = load_latest(target)
        assert back["step"] == 7
        # sibling node dir stays empty - no cross-node state bleed
        assert load_latest(checkpoint_dir_for("sushi", d)) is None
