"""Tests for SEAFile packaging with Materials provenance (pyslice_export_sea_file).

Verifies the agreed layout: simulation results in ``SEAFile.Simulations``,
the unit cell as a *Material* entry and the built structure as a *Sample*
entry in ``SEAFile.Materials``, the build record under
``Sample.Metadata.build``, database info under ``Metadata.Database``, and
the Sample's SEAID rooted at the Material's (the provenance relation) —
all surviving a ``.sea`` round-trip through sea-eco's generic loader.
"""
import json

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write as ase_write

pytest.importorskip("pySEA.sea_eco", reason="sea-eco not installed")

from pyslice.mcp.service import (
    BuildSlabInput,
    ExportSeaFileInput,
    PySliceService,
    RenderSignalInput,
    SetupMultisliceInput,
    TrajectoryOperation,
)


def _norm_id(value) -> str:
    """Normalize a SEAID string for comparison (Crockford O/0, hyphens)."""
    return str(value).replace("-", "").upper().replace("O", "0")


@pytest.fixture()
def service(tmp_path):
    """Service with an isolated workspace."""
    return PySliceService(workspace=tmp_path / "ws")


@pytest.fixture()
def packaged(service, tmp_path, monkeypatch):
    """Run a miniature pipeline and export a SEAFile; return its parts."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "diamond.cif"
    ase_write(str(path), bulk("C", "diamond", a=3.567))
    unit = service.load_structure(str(path), None, None)["handle"]
    slab = service.build_slab(BuildSlabInput(
        structure_handle=unit, indices=[1, 1, 0], layers=2, repeats=[2, 2], name="dia110"))
    sample = service.transform_trajectory(
        slab["handle"],
        [TrajectoryOperation(op="frozen_phonon", params={"n": 2, "sigma_A": 0.05, "seed": 7})],
        name="dia110-fp",
    )
    setup = service.setup_multislice(SetupMultisliceInput(
        trajectory_handle=sample["handle"], sampling_A=0.5, slice_thickness_A=1.26, force_cpu=True))
    run = service.run_multislice(setup["handle"], False)
    exported = service.export_sea_file(ExportSeaFileInput(
        filename="results/demo",
        signal_handles=[run["wf"]["handle"]],
        sample_handle=sample["handle"],
        material_handle=unit,
        name="demo package",
        material_name="C (diamond)",
    ))
    return service, run, exported


def test_sea_file_round_trip_materials_and_provenance(packaged):
    service, run, exported = packaged
    from pySEA.sea_eco.io import load as sea_load

    sea_file = sea_load(exported["sea_path"])
    names = [d.name for d in sea_file.Materials.datasets]
    assert names == ["C (diamond)", "Sample"]
    assert len(sea_file.Simulations) == 1

    material = sea_file.Materials.datasets[0]
    sample = sea_file.Materials.datasets[1]
    # provenance relation: Sample roots at the Material
    assert _norm_id(sample.Provenance.root) == _norm_id(material.Provenance)
    assert _norm_id(exported["material_seaid"]) == _norm_id(material.Provenance)

    # Material metadata: formula + box; Sample metadata: build record
    assert material.metadata.Material.formula == "C2"
    assert material.metadata.Material.kind == "Material"
    build = sample.metadata.build
    assert build.indices is not None
    operations = [json.loads(str(op)) for op in build.operations]
    assert operations[-1]["op"] == "frozen_phonon"

    # Atom-record format (sea-eco SignalSet layout): typed members sharing
    # the atom dimension, positions with a categorical component axis.
    assert type(sample).__name__ == "SignalSet"
    member_names = sample.get_dataset_names()
    assert member_names[:2] == ["positions", "element"]
    positions = sample["positions"]
    assert positions.data.shape[0] == 2  # frozen-phonon frames
    assert positions.data.shape[2] == 3
    component = positions._local_dimensions.dimensions[2]
    assert [str(v) for v in component.values] == ["x", "y", "z"]
    # component axis is structural: role-unassigned (neither nav nor det)
    assert 2 not in positions._local_dimensions.nav_dimensions
    assert 2 not in positions._local_dimensions.det_dimensions
    elements = sample["element"]
    assert set(str(e) for e in elements.data) == {"C"}
    assert len(elements.data) == positions.data.shape[1]


def test_simulation_signal_is_plain_and_calibrated(packaged):
    service, run, exported = packaged
    from pySEA.sea_eco.architecture.base_structure import Signal
    from pySEA.sea_eco.io import load as sea_load

    sea_file = sea_load(exported["sea_path"])
    simulation = sea_file.Simulations.datasets[0]
    assert type(simulation).__name__ == "Signal"  # readable without PySlice
    wf = service._get(run["wf"]["handle"])
    np.testing.assert_allclose(np.asarray(simulation.data), wf.data)
    dim_names = [d.name for d in simulation._local_dimensions.dimensions]
    assert dim_names == ["probe", "time", "kx", "ky", "layer"]


def test_render_signal_uses_sea_eco(packaged):
    service, run, exported = packaged
    result = service.render_signal(RenderSignalInput(handle=run["wf"]["handle"], filename="wf.png"))
    assert result["renderer"] == "sea-eco matplotlib"
    assert result["artifact_path"].endswith("wf.png")


def test_database_source_flows_into_material_metadata(service, tmp_path, monkeypatch):
    import pyslice.io.databases as databases

    monkeypatch.setattr(
        databases, "_http_get",
        lambda url, headers=None, timeout=30.0: (
            b"data_test\n_symmetry_space_group_name_H-M 'P 1'\n"
            b"_cell_length_a 3.0\n_cell_length_b 3.0\n_cell_length_c 3.0\n"
            b"_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            b"loop_\n_atom_site_type_symbol\n_atom_site_label\n"
            b"_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n"
            b"Si Si1 0 0 0 1\n"
        ),
    )
    fetched = service.fetch_structure("cod", "12345", None, True, None, None)
    signal = service._trajectory_to_material_set(
        service._get(fetched["handle"]), name="Material", kind="Material",
        source=service._source_info[fetched["handle"]],
    )
    assert signal.metadata.Database.provider == "cod"
    assert signal.metadata.Database.entry_id == "12345"
