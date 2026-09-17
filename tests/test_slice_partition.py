"""The slice partition, and what happens to atoms that sit on its boundaries.

``Potential._slice_bounds`` defines which atoms contribute to which slice. If
consecutive slices do not share an edge exactly, an atom on a boundary is
either dropped from the potential or counted in two slices, and neither is
reported. The result is a simulation that runs to completion and is wrong.

These are unit tests on the partition itself plus one end-to-end check, because
the partition property is the thing that must hold and the physics consequence
is the thing that must be seen to follow from it.
"""

import numpy as np
import pytest

from pyslice import Potential
from pyslice.backend import make_backend


def grid(height, n_slices, n_lateral=32, lateral=5.0):
    """Axes for a cell of the given height sliced into ``n_slices``.

    ``zs`` follows ``grid_from_trajectory``: slice *lower edges* starting at
    zero, not centres. Slice ``i`` then spans ``[zs[i] - dz/2, zs[i] + dz/2)``,
    which leaves the first slice half-width and the last one and a half, and
    covers ``[0, height]`` exactly. That asymmetry is upstream's convention and
    is preserved here -- this module is about the edges being *shared*, not
    about where they sit.
    """
    spacing = height / n_slices
    return (
        np.linspace(0.0, lateral, n_lateral, endpoint=False),
        np.linspace(0.0, lateral, n_lateral, endpoint=False),
        np.arange(n_slices) * spacing,
    )


def build(xs, ys, zs, z_positions):
    positions = np.zeros((len(z_positions), 3))
    positions[:, 0] = 2.5
    positions[:, 1] = 2.5
    positions[:, 2] = z_positions
    return Potential(xs, ys, zs, positions, ["C"] * len(z_positions),
                     backend=make_backend())


@pytest.mark.parametrize("height", [7.34, 20.0, 33.3, 100.0])
@pytest.mark.parametrize("n_slices", [3, 7, 15, 16, 31, 64, 101, 128])
def test_consecutive_slices_share_an_edge_exactly(height, n_slices):
    xs, ys, zs = grid(height, n_slices)
    potential = build(xs, ys, zs, [height / 2])
    for i in range(n_slices - 1):
        assert potential._slice_bounds(i)[1] == potential._slice_bounds(i + 1)[0], (
            f"slice {i} ends where slice {i + 1} does not begin; "
            "an atom between the two is dropped or double-counted"
        )


def test_the_partition_covers_the_whole_cell():
    height, n_slices = 7.34, 16
    xs, ys, zs = grid(height, n_slices)
    potential = build(xs, ys, zs, [height / 2])
    assert potential._slice_bounds(0)[0] == 0.0
    assert potential._slice_bounds(n_slices - 1)[1] == pytest.approx(height)


def test_every_atom_lands_in_exactly_one_slice():
    # Atoms are placed *on* the slice edges, which is the case the shipped
    # bounds got wrong. Membership uses the same comparison the potential
    # does: lo <= z < hi.
    height, n_slices = 7.34, 16
    xs, ys, zs = grid(height, n_slices)
    spacing = height / n_slices
    edges = np.asarray(zs) - spacing / 2.0
    edges = edges[edges > 0.0]
    potential = build(xs, ys, zs, edges)

    bounds = [potential._slice_bounds(i) for i in range(n_slices)]
    for z in edges:
        owners = [i for i, (lo, hi) in enumerate(bounds) if lo <= z < hi]
        assert len(owners) == 1, f"z={z!r} belongs to slices {owners}, not exactly one"


@pytest.mark.parametrize("height,n_slices", [(33.3, 64), (47.9, 31), (100.0, 101)])
def test_atom_planes_on_boundaries_are_counted_once(height, n_slices):
    # The physical consequence, and the reason this is not a pedantic point
    # about floating point. The volume integral of the projected potential is
    # the sum of the atoms' own integrals and does not depend on where along
    # the beam they sit, so nudging a plane off a boundary must not change it.
    #
    # Under the previous bounds it changes a lot. These three grids give
    # 1.44x, 1.37x and 1.32x the correct potential, because atoms landing in
    # an overlap are built into two slices. Grids whose arithmetic happens
    # to be exact -- 6.0 A in 12 slices, say -- show nothing, which is what
    # makes the failure hard to notice in practice.
    xs, ys, zs = grid(height, n_slices)
    spacing = height / n_slices
    on_edge = np.asarray(zs) - spacing / 2.0
    on_edge = on_edge[on_edge > 0.0]

    exact = build(xs, ys, zs, on_edge)
    exact.build()
    nudged = build(xs, ys, zs, on_edge + 0.01 * spacing)
    nudged.build()

    total_exact = float(exact.to_numpy().sum())
    total_nudged = float(nudged.to_numpy().sum())
    assert total_exact == pytest.approx(total_nudged, rel=1e-9), (
        f"{len(on_edge)} atoms on slice boundaries of a {height} A cell in "
        f"{n_slices} slices integrate to {total_exact:.6e}; the same atoms "
        f"nudged off the boundaries give {total_nudged:.6e}, a factor of "
        f"{total_exact / total_nudged:.5f}. Atoms are being dropped or "
        "double-counted."
    )


@pytest.mark.parametrize("height,n_slices", [(33.3, 64), (47.9, 31), (100.0, 101)])
def test_atom_planes_on_upper_boundaries_are_not_dropped(height, n_slices):
    # The other half of the defect, and the half that loses signal rather than
    # inventing it. Atoms on a slice's *lower* edge land in an overlap and are
    # built twice; atoms on its *upper* edge land in a gap and vanish. These
    # three grids lose 17%, 3% and 24% of the projected potential under the
    # previous bounds. Both halves are real, and the losing half is the
    # harder one to notice -- a dimmer image looks like a thinner sample.
    xs, ys, zs = grid(height, n_slices)
    spacing = height / n_slices
    on_edge = np.asarray(zs) + spacing / 2.0
    on_edge = on_edge[on_edge < height]

    exact = build(xs, ys, zs, on_edge)
    exact.build()
    nudged = build(xs, ys, zs, on_edge - 0.01 * spacing)
    nudged.build()

    total_exact = float(exact.to_numpy().sum())
    total_nudged = float(nudged.to_numpy().sum())
    assert total_exact == pytest.approx(total_nudged, rel=1e-9), (
        f"{len(on_edge)} atoms on upper slice boundaries of a {height} A cell "
        f"in {n_slices} slices integrate to {total_exact:.6e}; the same atoms "
        f"nudged off the boundaries give {total_nudged:.6e}, a factor of "
        f"{total_exact / total_nudged:.5f}. Atoms are being dropped."
    )


@pytest.mark.parametrize("z", [-1e-16, -1e-15, -5e-16])
def test_a_tiny_negative_coordinate_is_not_dropped(z):
    # np.mod(z, period) returns exactly period for z in roughly
    # (-ulp(period)/2, 0), and the top bound is half-open, so the atom lands in
    # no slice at all and its potential silently disappears. Coordinates like
    # this arrive from a wrap or a subtraction, not from a file, which is why
    # nothing upstream notices.
    height, n_slices = 20.0, 16
    xs, ys, zs = grid(height, n_slices)

    dropped = build(xs, ys, zs, np.array([z]))
    dropped.build()
    reference = build(xs, ys, zs, np.array([0.0]))
    reference.build()

    total = float(dropped.to_numpy().sum())
    expected = float(reference.to_numpy().sum())
    assert total == pytest.approx(expected, rel=1e-9), (
        f"an atom at z={z:g} integrates to {total:.6e} where the same atom at "
        f"z=0 gives {expected:.6e}; np.mod has folded it onto the exclusive "
        f"top edge and it belongs to no slice."
    )
