"""Tests for pyslice.io.databases, which retrieves through sea-eco's importers.

sea-eco's single network seam (``pySEA.sea_eco.databases._http._open``) is
stubbed with each API's response shape; set ``PYSLICE_DB_LIVE_TESTS=1`` to
also run the keyless live COD round-trip on a machine with open egress.
"""
import io
import json
import os
import urllib.error
import urllib.parse

import numpy as np
import pytest

from pyslice.data import seashell

pytestmark = pytest.mark.skipif(not seashell.sea_available, reason="retrieval needs sea-eco")

from pyslice.io.databases import (  # noqa: E402
    DatabaseError,
    _atomic_structure_to_cif,
    database_title,
    fetch_cif,
    load_structure_from_database,
    search_structures,
)

MP_SILICON = {
    "material_id": "mp-149",
    "formula_pretty": "Si",
    "symmetry": {"symbol": "Fd-3m", "number": 227, "crystal_system": "Cubic"},
    "structure": {
        "lattice": {"matrix": [[3.3489, 0.0, 1.9335], [1.1163, 3.1574, 1.9335], [0.0, 0.0, 3.867]]},
        "sites": [
            {"species": [{"element": "Si", "occu": 1}], "abc": [0.875, 0.875, 0.875]},
            {"species": [{"element": "Si", "occu": 1}], "abc": [0.125, 0.125, 0.125]},
        ],
    },
}
NACL_CIF = """data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0 0 0
Cl1 Cl 0.5 0.5 0.5
"""


@pytest.fixture
def web(monkeypatch, tmp_path):
    """Serve stubbed responses to sea-eco and keep its cache in ``tmp_path``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture.
    tmp_path : pathlib.Path
        sea-eco cache root.

    Returns
    -------
    dict
        ``routes`` (URL without query -> body) and ``requests`` (URL,
        headers) the stub received.
    """
    from pySEA.sea_eco.credentials import SeaEcoSecrets
    from pySEA.sea_eco.databases import _base, _http

    state = {"routes": {}, "requests": []}

    def serve(request, timeout):
        """Answer from ``routes``; 404 otherwise."""
        state["requests"].append((request.full_url, dict(request.header_items())))
        body = state["routes"].get(request.full_url.split("?")[0])
        if body is None:
            raise urllib.error.HTTPError(request.full_url, 404, "stub", {}, None)
        return io.BytesIO(body if isinstance(body, bytes) else json.dumps(body).encode())

    monkeypatch.setattr(_http, "_open", serve)
    monkeypatch.setattr(_base, "default_cache_directory", lambda: tmp_path / "sea-cache")
    monkeypatch.setattr(SeaEcoSecrets, "get_api_token", classmethod(lambda cls, profile: None))
    monkeypatch.delenv("PYSLICE_MP_API_KEY", raising=False)
    monkeypatch.setenv("MP_API_KEY", "test-key")
    return state


def _query(url):
    """Return a URL's query as a dict."""
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


def test_search_requires_a_filter():
    with pytest.raises(ValueError, match="formula"):
        search_structures("cod")


def test_file_repositories_are_not_structure_providers():
    with pytest.raises(ValueError, match="file repository"):
        search_structures("zenodo", formula="Si")


def test_mp_without_a_key_says_where_to_get_one(web, monkeypatch):
    monkeypatch.delenv("MP_API_KEY")
    with pytest.raises(DatabaseError, match="materialsproject.org/api"):
        search_structures("mp", formula="Si")


def test_mp_search_keeps_pyslice_keys_and_sorts_stable_first(web):
    web["routes"]["https://api.materialsproject.org/materials/summary/"] = {"data": [
        {"material_id": "mp-2", "formula_pretty": "Si", "symmetry": {"symbol": "P1"}, "energy_above_hull": 0.4},
        {"material_id": "mp-149", "formula_pretty": "Si", "symmetry": {"symbol": "Fd-3m"}, "energy_above_hull": 0.0},
    ]}
    entries = search_structures("mp", formula="Si", limit=5)

    url, headers = web["requests"][-1]
    assert headers["X-api-key"] == "test-key"
    assert _query(url)["formula"] == "Si"
    assert [e["id"] for e in entries] == ["mp-149", "mp-2"]
    assert entries[0]["provider"] == "mp" and entries[0]["spacegroup"] == "Fd-3m"


def test_explicit_api_key_wins(web):
    web["routes"]["https://api.materialsproject.org/materials/summary/"] = {"data": []}
    search_structures("mp", formula="Si", api_key="explicit")
    assert web["requests"][-1][1]["X-api-key"] == "explicit"


def test_cod_search_uses_hill_formula(web):
    web["routes"]["https://www.crystallography.net/cod/result"] = [
        {"file": 12345, "formula": "- O2 Si -", "sg": "P 1", "a": 4.9, "b": 4.9, "c": 5.4},
    ]
    entries = search_structures("cod", formula="SiO2", elements=["Si", "O"], limit=3)

    query = _query(web["requests"][-1][0])
    assert query["formula"] == "O2 Si" and query["el1"] == "Si" and query["el2"] == "O"
    assert entries[0]["id"] == "12345" and entries[0]["formula"] == "O2 Si"


def test_fetch_cif_cod_copies_the_published_file(web, tmp_path):
    web["routes"]["https://www.crystallography.net/cod/12345.cif"] = NACL_CIF.encode()
    path = fetch_cif("cod", "12345", output_dir=tmp_path / "out")
    assert path.name == "cod_12345.cif"
    assert path.read_text() == NACL_CIF


def test_fetch_cif_cod_rejects_non_numeric_id(web, tmp_path):
    with pytest.raises(DatabaseError, match="numeric"):
        fetch_cif("cod", "mp-149", output_dir=tmp_path)


def test_fetch_cif_mp_renders_a_loadable_p1_cif(web, tmp_path):
    web["routes"]["https://api.materialsproject.org/materials/summary/"] = {"data": [MP_SILICON]}
    path = fetch_cif("mp", "mp-149", output_dir=tmp_path / "out")
    assert path.name == "mp_mp_149.cif"
    assert "'P 1'" in path.read_text()

    from pyslice.io.loader import Loader

    trajectory = Loader(filename=str(path)).load()
    assert trajectory.n_atoms == 2
    assert sorted(set(trajectory.atom_types)) == ["Si"]
    matrix = np.asarray(MP_SILICON["structure"]["lattice"]["matrix"])
    assert np.isclose(abs(np.linalg.det(trajectory.box_matrix)), abs(np.linalg.det(matrix)), rtol=1e-4)


def test_optimade_entry_loads_as_a_trajectory(web, tmp_path):
    web["routes"]["https://oqmd.org/optimade/v1/structures/oqmd-1"] = {"data": {
        "id": "oqmd-1",
        "attributes": {
            "lattice_vectors": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            "cartesian_site_positions": [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
            "species_at_sites": ["Cs", "Cl"],
            "dimension_types": [1, 1, 1],
        },
    }}
    trajectory = load_structure_from_database("oqmd", "oqmd-1", output_dir=tmp_path)
    assert sorted(trajectory.atom_types) == ["Cl", "Cs"]
    assert database_title("oqmd") == "Open Quantum Materials Database"


def test_structure_without_a_3d_cell_is_refused(tmp_path):
    from pySEA.sea_eco.signal_containers import build_atomic_structure

    slab = build_atomic_structure(
        [[0.0, 0.0, 0.0]], ["Mo"], [[3.16, 0.0, 0.0], [-1.58, 2.74, 0.0], [0.0, 0.0, 0.0]],
        periodic=[True, True, False],
    )
    with pytest.raises(DatabaseError, match="3-D cell"):
        _atomic_structure_to_cif(slab, "slab")


@pytest.mark.skipif(not os.environ.get("PYSLICE_DB_LIVE_TESTS"), reason="set PYSLICE_DB_LIVE_TESTS=1 for live COD access")
def test_live_cod_round_trip(tmp_path):
    entries = search_structures("cod", elements=["Ti", "O"], limit=3)
    assert entries, "COD returned no Ti-O entries"
    trajectory = load_structure_from_database("cod", entries[0]["id"], output_dir=tmp_path)
    assert trajectory.n_atoms > 0
