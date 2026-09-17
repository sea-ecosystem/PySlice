"""Selecting the arithmetic precision of the torch backend.

The default is double everywhere except MPS, and this module's first duty is
to keep it that way: precision is a property of the physics engine, so a
change to it should be something a caller asked for, never something that
happened.

The reason to offer single at all is that on consumer and workstation Ampere
the double-precision rate is a small fraction of single -- a 3.8x end-to-end
difference on an RTX A5000 -- while the answers agree to far below the spread
between frozen-phonon seeds. That is a trade worth being able to make
explicitly, which is what these tests pin down.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from pyslice.backend import TorchBackend, make_backend


@pytest.fixture
def no_precision_env(monkeypatch):
    monkeypatch.delenv("PYSLICE_PRECISION", raising=False)
    monkeypatch.delenv("PYSLICE_BACKEND", raising=False)


def test_the_default_is_double(no_precision_env):
    backend = TorchBackend(device="cpu")
    assert backend.float_dtype is torch.float64
    assert backend.complex_dtype is torch.complex128


def test_single_precision_can_be_requested(no_precision_env):
    backend = TorchBackend(device="cpu", precision="single")
    assert backend.float_dtype is torch.float32
    assert backend.complex_dtype is torch.complex64


def test_double_precision_can_be_requested(no_precision_env):
    backend = TorchBackend(device="cpu", precision="double")
    assert backend.float_dtype is torch.float64


def test_the_environment_variable_selects_precision(monkeypatch):
    monkeypatch.delenv("PYSLICE_BACKEND", raising=False)
    monkeypatch.setenv("PYSLICE_PRECISION", "single")
    assert TorchBackend(device="cpu").float_dtype is torch.float32
    monkeypatch.setenv("PYSLICE_PRECISION", "double")
    assert TorchBackend(device="cpu").float_dtype is torch.float64


def test_the_argument_beats_the_environment(monkeypatch):
    # Otherwise a caller who asked for double could silently get single
    # because of a variable set elsewhere in the process.
    monkeypatch.delenv("PYSLICE_BACKEND", raising=False)
    monkeypatch.setenv("PYSLICE_PRECISION", "single")
    assert TorchBackend(device="cpu", precision="double").float_dtype is torch.float64


def test_an_unknown_precision_is_rejected(no_precision_env):
    with pytest.raises(ValueError, match="precision must be one of"):
        TorchBackend(device="cpu", precision="float16")
    with pytest.raises(ValueError, match="precision must be one of"):
        TorchBackend(device="cpu", precision="half")


def test_an_unknown_precision_in_the_environment_is_rejected(monkeypatch):
    monkeypatch.delenv("PYSLICE_BACKEND", raising=False)
    monkeypatch.setenv("PYSLICE_PRECISION", "quad")
    with pytest.raises(ValueError, match="precision must be one of"):
        TorchBackend(device="cpu")


def test_make_backend_passes_precision_through(no_precision_env):
    assert make_backend(device="cpu", precision="single").float_dtype is torch.float32
    assert make_backend(device="cpu").float_dtype is torch.float64


def test_the_numpy_backend_ignores_precision(monkeypatch):
    monkeypatch.setenv("PYSLICE_BACKEND", "numpy")
    backend = make_backend(precision="single")
    assert type(backend).__name__ == "NumpyBackend"


def test_single_precision_arrays_round_trip(no_precision_env):
    # The dtypes have to reach the arrays, not just the attributes.
    backend = TorchBackend(device="cpu", precision="single")
    array = backend.zeros((4, 4), dtype=backend.complex_dtype)
    assert array.dtype is torch.complex64


def test_whitespace_around_the_environment_value_is_tolerated(monkeypatch):
    monkeypatch.delenv("PYSLICE_BACKEND", raising=False)
    monkeypatch.setenv("PYSLICE_PRECISION", "  single\n")
    assert TorchBackend(device="cpu").float_dtype is torch.float32


def test_the_slice_cache_is_keyed_on_dtype(tmp_path, monkeypatch):
    # Precision became selectable, so one cache_dir can be visited by a single
    # run and a double run. Without a dtype in the name the second silently
    # reads the first's float32 slices back and reports them as double.
    monkeypatch.delenv("PYSLICE_BACKEND", raising=False)
    monkeypatch.delenv("PYSLICE_PRECISION", raising=False)
    import numpy as np
    from pyslice import Potential

    xs = np.linspace(0.0, 5.0, 24, endpoint=False)
    positions = np.array([[2.0, 2.5, 1.0], [3.0, 2.5, 2.0]])
    zs = np.arange(8) * 0.5

    totals = {}
    for precision in ("single", "double"):
        backend = make_backend(device="cpu", precision=precision)
        potential = Potential(xs, xs, zs, positions, ["C", "C"],
                              backend=backend, cache_dir=tmp_path, frame_idx=0)
        potential.build()
        totals[precision] = float(potential.to_numpy().sum())

    names = sorted(p.name for p in tmp_path.glob("*.npy"))
    assert any("float32" in n for n in names), names
    assert any("float64" in n for n in names), names

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    backend = make_backend(device="cpu", precision="double")
    potential = Potential(xs, xs, zs, positions, ["C", "C"],
                          backend=backend, cache_dir=fresh, frame_idx=0)
    potential.build()
    assert float(potential.to_numpy().sum()) == pytest.approx(
        totals["double"], rel=1e-12
    ), "the double run read a single-precision cache"
