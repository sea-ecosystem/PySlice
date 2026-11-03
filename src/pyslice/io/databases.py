"""Search crystal-structure databases and retrieve CIF files for PySlice.

PySlice keeps no database client of its own: every request goes through
sea-eco's importers, ``pySEA.sea_eco.databases``, reached only through
:func:`pyslice.data.seashell.sea_databases`. That backend serves the
Materials Project (API key), the Crystallography Open Database (keyless), and
the OPTIMADE providers OQMD, NOMAD, JARVIS-DFT and Alexandria (keyless), and
caches what it fetches. This module turns its entries into CIF files that
:class:`pyslice.io.loader.Loader` reads, so retrieval needs
``pip install 'pyslice[sea]'`` (and ASE, which PySlice already requires).

The Materials Project key is read by sea-eco: ``api_key=``, then sea-eco's
credential store (``SeaEcoSecrets.set_api_token("materials_project", key)``),
then ``MP_API_KEY`` or ``PYSLICE_MP_API_KEY``. Get one free at
https://next-gen.materialsproject.org/api.
"""
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np

from ..data import seashell

logger = logging.getLogger(__name__)

STRUCTURE_PROVIDERS = ("mp", "cod", "oqmd", "nomad", "jarvis", "alexandria")
Provider = Literal["mp", "cod", "oqmd", "nomad", "jarvis", "alexandria"]

# Angstrom per stored length unit; sea-eco structures store their units on the
# scalar SignalQuantities (signal-containers CONT-6) and are written in Å.
_ANGSTROM_PER_UNIT = {"å": 1.0, "Å": 1.0, "a": 1.0, "angstrom": 1.0, "nm": 10.0, "pm": 0.01}


class DatabaseError(RuntimeError):
    """Raised when a structure-database request cannot be completed.

    Wraps sea-eco's ``DatabaseError`` (network failures, authentication
    problems, missing entries, malformed responses) so PySlice callers catch
    one PySlice type; the message is sea-eco's and names the fix.

    Parameters
    ----------
    message : str
        Human-readable description including the suggested fix.

    Attributes
    ----------
    args : tuple
        Standard exception arguments; ``args[0]`` is the message.

    Methods
    -------
    (inherits all behavior from RuntimeError)

    Raises
    ------
    (never raises on construction)

    See Also
    --------
    search_structures : Query a database for matching entries.
    fetch_cif : Download one entry as a CIF file.

    Notes
    -----
    A dedicated type lets callers (e.g. the MCP service layer) distinguish
    database problems from programming errors. The sea-eco exception is kept
    as ``__cause__``.

    Examples
    --------
    >>> raise DatabaseError("COD returned no entries for formula 'Xx'")
    Traceback (most recent call last):
    ...
    pyslice.io.databases.DatabaseError: COD returned no entries for formula 'Xx'
    """


def _client(provider: str, api_key: Optional[str]) -> Any:
    """Return sea-eco's client for a structure database.

    Parameters
    ----------
    provider : str
        sea-eco database name, one of :data:`STRUCTURE_PROVIDERS` or any
        other structure database registered with sea-eco.
    api_key : str | None
        Materials Project key; only used for ``provider="mp"``.

    Returns
    -------
    pySEA.sea_eco.databases.Database
        The client.

    Raises
    ------
    ImportError
        If sea-eco (with its ``databases``) is not installed.
    ValueError
        If the provider is unknown to sea-eco or is a file repository.
    """
    databases = seashell.sea_databases()
    if provider == "mp" and api_key:
        return databases.MaterialsProject(api_key=api_key)
    client = databases.get_database(provider)
    if client.kind != "structure":
        raise ValueError(
            f"{provider!r} is a file repository, not a structure database; PySlice retrieves "
            f"structures from {', '.join(STRUCTURE_PROVIDERS)}."
        )
    return client


def _call(action: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a sea-eco request, re-raising its failures as :class:`DatabaseError`.

    Parameters
    ----------
    action : callable
        Bound client method.
    *args, **kwargs : Any
        Its arguments.

    Returns
    -------
    Any
        Whatever ``action`` returns.

    Raises
    ------
    DatabaseError
        If sea-eco raised its ``DatabaseError``.
    """
    sea_error = seashell.sea_databases().DatabaseError
    try:
        return action(*args, **kwargs)
    except sea_error as exc:
        raise DatabaseError(str(exc)) from exc


def database_title(provider: str) -> str:
    """Return the human-readable name of a database, for provenance records.

    Parameters
    ----------
    provider : str
        sea-eco database name, e.g. ``"oqmd"``.

    Returns
    -------
    str
        For example ``"Open Quantum Materials Database"``.

    Raises
    ------
    ImportError
        If sea-eco is not installed.
    ValueError
        If the provider is unknown.

    Examples
    --------
    >>> database_title("cod")  # doctest: +SKIP
    'Crystallography Open Database'
    """
    return seashell.sea_databases().get_database(provider).title


def search_structures(
    provider: Provider,
    formula: Optional[str] = None,
    elements: Optional[Sequence[str]] = None,
    limit: int = 20,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search a crystal-structure database for matching entries.

    Queries one of sea-eco's structure databases by chemical formula and/or
    element set, returning lightweight entry summaries whose ``id`` values
    feed :func:`fetch_cif`.

    Parameters
    ----------
    provider : {"mp", "cod", "oqmd", "nomad", "jarvis", "alexandria"}
        Database to query.
    formula : str | None, optional
        Chemical formula, e.g. ``"SiO2"`` or ``"BaTiO3"``. MP accepts
        wildcards such as ``"Si*"``; OPTIMADE providers match the reduced
        formula.
    elements : Sequence[str] | None, optional
        Element symbols every match must contain, e.g. ``("Ga", "N")``.
    limit : int, optional
        Maximum number of entries to return, by default 20.
    api_key : str | None, optional
        Materials Project API key; looked up by sea-eco when None. Ignored
        for the other databases.

    Returns
    -------
    list[dict]
        sea-eco's summaries (``database``, ``id``, ``formula``,
        ``space_group`` plus per-database fields such as MP's
        ``energy_above_hull_eV``, sorted most stable first), each also
        carrying PySlice's ``provider`` and ``spacegroup`` keys.

    Raises
    ------
    ValueError
        If neither ``formula`` nor ``elements`` is given, or the provider is
        unknown or not a structure database.
    DatabaseError
        If the request fails (network, authentication, malformed response).
    ImportError
        If sea-eco is not installed.

    See Also
    --------
    fetch_cif : Download one of the returned entries as a CIF file.
    load_structure_from_database : Fetch and load in one step.

    Notes
    -----
    MP entries are computed (DFT-relaxed) structures; COD entries are
    experimental crystal structures from the literature; OQMD, NOMAD, JARVIS
    and Alexandria are computed. ``elements`` means "contains all of these"
    for every database; a formula pins the composition exactly.

    Examples
    --------
    >>> entries = search_structures("cod", formula="C")  # doctest: +SKIP
    >>> entries[0]["id"]  # doctest: +SKIP
    '9008569'
    """
    if not formula and not elements:
        raise ValueError("Provide a formula (e.g. 'SiO2') and/or an elements list (e.g. ['Ga', 'N']).")
    client = _client(provider, api_key)
    entries = _call(client.search, formula=formula, elements=elements, limit=limit)
    return [{**entry, "provider": provider, "spacegroup": entry.get("space_group")} for entry in entries]


def _length_factor(signal: Any) -> float:
    """Return angstrom per stored unit of a structure's length Signal.

    Parameters
    ----------
    signal : Signal
        ``atoms.position`` or ``cell.cell``.

    Returns
    -------
    float
        Conversion factor to Å.

    Raises
    ------
    DatabaseError
        If the units are not a recognised length unit.
    """
    quantities = getattr(signal, "signal_quantities", None)
    units = getattr(quantities[0], "units", "") if quantities is not None and len(quantities) else ""
    try:
        return _ANGSTROM_PER_UNIT[str(units or "Å").strip().lower()]
    except KeyError:
        raise DatabaseError(f"Structure lengths are in {units!r}; expected Å, nm or pm.") from None


def _atomic_structure_to_cif(structure: Any, data_name: str) -> str:
    """Render a sea-eco ``atomic-structure`` collection as a P1 CIF.

    Every site is written explicitly with its occupancy, so no symmetry is
    lost by using P1 and :class:`pyslice.io.loader.Loader` (via ASE) reads it
    as it read the CIFs PySlice wrote before retrieval moved to sea-eco.

    Parameters
    ----------
    structure : SignalCollection
        Static ``atomic-structure`` collection from sea-eco.
    data_name : str
        CIF data-block name.

    Returns
    -------
    str
        Complete CIF file content.

    Raises
    ------
    DatabaseError
        If the structure has no full three-dimensional cell (an entry that
        is not periodic in some direction cannot be written as a CIF).

    Notes
    -----
    Positions convert to fractional coordinates with the inverse cell; cell
    parameters come from the vector lengths and angles. A disordered site
    arrives from sea-eco as one atom per species with ``atoms.occupancy``,
    and is written that way.
    """
    atoms, cell_set = structure["atoms"], structure["cell"]
    cell = np.asarray(cell_set["cell"].data[:], dtype=float) * _length_factor(cell_set["cell"])
    positions = np.asarray(atoms["position"].data[:], dtype=float) * _length_factor(atoms["position"])
    elements = [str(element) for element in atoms["element"].data[:]]
    members = [member.name for member in atoms]
    occupancy = np.asarray(atoms["occupancy"].data[:], dtype=float) if "occupancy" in members else np.ones(len(elements))
    if abs(np.linalg.det(cell)) < 1e-9:
        raise DatabaseError(
            f"{data_name} has no full 3-D cell (it is not periodic in every direction), so it "
            "cannot be written as a CIF; open it with sea-eco and build a slab instead."
        )
    fractional = positions @ np.linalg.inv(cell)
    lengths = np.linalg.norm(cell, axis=1)

    def angle(i: int, j: int) -> float:
        """Return the angle between cell vectors ``i`` and ``j`` in degrees."""
        cosine = np.dot(cell[i], cell[j]) / (lengths[i] * lengths[j])
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    lines = [
        f"data_{data_name}",
        "_symmetry_space_group_name_H-M   'P 1'",
        "_symmetry_Int_Tables_number      1",
        f"_cell_length_a     {lengths[0]:.6f}",
        f"_cell_length_b     {lengths[1]:.6f}",
        f"_cell_length_c     {lengths[2]:.6f}",
        f"_cell_angle_alpha  {angle(1, 2):.6f}",
        f"_cell_angle_beta   {angle(0, 2):.6f}",
        f"_cell_angle_gamma  {angle(0, 1):.6f}",
        "loop_",
        "_atom_site_type_symbol",
        "_atom_site_label",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    label_counts: Dict[str, int] = {}
    for element, (fx, fy, fz), occ in zip(elements, fractional, occupancy):
        label_counts[element] = label_counts.get(element, 0) + 1
        lines.append(f"{element} {element}{label_counts[element]} {fx:.6f} {fy:.6f} {fz:.6f} {occ:.4f}")
    return "\n".join(lines) + "\n"


def fetch_cif(
    provider: Provider,
    entry_id: str,
    output_dir: str | Path = ".",
    filename: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Path:
    """Download one database entry as a CIF file.

    For COD the published CIF is copied verbatim from sea-eco's cache; for
    every other database sea-eco's ``atomic-structure`` import is rendered as
    a P1 CIF (all sites explicit). The resulting file loads with
    ``Loader(filename=str(path)).load()``.

    Parameters
    ----------
    provider : {"mp", "cod", "oqmd", "nomad", "jarvis", "alexandria"}
        Database the entry comes from.
    entry_id : str
        Entry identifier, e.g. ``"mp-149"`` (MP) or ``"1010939"`` (COD), as
        returned by :func:`search_structures`.
    output_dir : str | pathlib.Path, optional
        Directory to write the CIF into (created if missing), by default the
        current directory.
    filename : str | None, optional
        Output filename; defaults to ``<provider>_<entry_id>.cif`` with
        characters other than letters, digits and ``.`` replaced by ``_``.
    api_key : str | None, optional
        Materials Project API key; looked up by sea-eco when None.

    Returns
    -------
    pathlib.Path
        Path of the written CIF file.

    Raises
    ------
    ValueError
        If the provider is unknown or not a structure database.
    DatabaseError
        If the entry cannot be retrieved, or has no full 3-D cell.
    ImportError
        If sea-eco is not installed.

    See Also
    --------
    search_structures : Find entry ids to fetch.
    load_structure_from_database : Fetch and load in one step.

    Notes
    -----
    sea-eco keeps its own cache of every fetched entry
    (``pySEA.sea_eco.databases.default_cache_directory()``), so fetching the
    same entry again needs no network.

    Examples
    --------
    >>> path = fetch_cif("cod", "1010939", output_dir="structures")  # doctest: +SKIP
    >>> path.name  # doctest: +SKIP
    'cod_1010939.cif'
    """
    client = _client(provider, api_key)
    entry_id = str(entry_id).strip()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9.]", "_", entry_id)
    path = output_dir / (filename or f"{provider}_{stem}.cif")
    if provider == "cod":
        shutil.copyfile(_call(client.fetch, entry_id), path)
    else:
        structure = _call(client.load, entry_id)
        path.write_text(_atomic_structure_to_cif(structure, stem), encoding="utf-8")
    logger.info("Wrote %s entry %s to %s", provider, entry_id, path)
    return path


def load_structure_from_database(
    provider: Provider,
    entry_id: str,
    output_dir: str | Path = ".",
    api_key: Optional[str] = None,
    **loader_kwargs: Any,
):
    """Fetch a database entry and load it as a PySlice ``Trajectory``.

    Convenience wrapper chaining :func:`fetch_cif` and
    :class:`pyslice.io.loader.Loader`, so a simulation can start from a
    database id in one call.

    Parameters
    ----------
    provider : {"mp", "cod", "oqmd", "nomad", "jarvis", "alexandria"}
        Database the entry comes from.
    entry_id : str
        Entry identifier, e.g. ``"mp-149"`` or ``"1010939"``.
    output_dir : str | pathlib.Path, optional
        Directory the CIF (and Loader ``.npy`` cache) is written into.
    api_key : str | None, optional
        Materials Project API key; looked up by sea-eco when None.
    **loader_kwargs : Any
        Extra keyword arguments forwarded to
        :class:`pyslice.io.loader.Loader` (e.g. ``timestep=``).

    Returns
    -------
    pyslice.multislice.trajectory.Trajectory
        Single-frame trajectory of the retrieved structure with
        element-symbol ``atom_types``.

    Raises
    ------
    ValueError
        If the provider is unknown.
    DatabaseError
        If the entry cannot be retrieved.
    ImportError
        If sea-eco is not installed.

    See Also
    --------
    search_structures : Find entry ids.
    fetch_cif : Just download the CIF without loading.

    Examples
    --------
    >>> traj = load_structure_from_database("cod", "1010939")  # doctest: +SKIP
    >>> traj.n_atoms  # doctest: +SKIP
    8
    """
    from .loader import Loader

    cif_path = fetch_cif(provider, entry_id, output_dir=output_dir, api_key=api_key)
    return Loader(filename=str(cif_path), **loader_kwargs).load()
