"""Service layer for the PySlice MCP server.

Holds the stateful :class:`PySliceService` used by every MCP tool, plus the
pydantic input models. The service keeps live PySlice objects (trajectories,
calculators, simulation results) in a handle registry so multi-step agent
workflows — search a database, load a structure, suggest parameters, run
multislice, post-process, export ``.sea`` — can pass objects between tool
calls without re-serializing them. Heavy PySlice modules are imported lazily
inside methods so the server starts without torch/OVITO present.

The layout mirrors ``pySEA.sea_eco.mcp.service`` (the ecosystem's reference
MCP implementation).
"""
from __future__ import annotations

import json
import math
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResponseFormat(str, Enum):
    """Supported MCP response formats.

    Markdown responses are compact and human-scannable; JSON responses carry
    the full machine-readable payload.

    Attributes
    ----------
    MARKDOWN : str
        Render the payload as simple Markdown.
    JSON : str
        Render the payload as indented JSON.

    Methods
    -------
    (inherits enum behavior)

    See Also
    --------
    PySliceService.format_response : Applies the selected format.
    """

    MARKDOWN = "markdown"
    JSON = "json"


class SimulationGoal(str, Enum):
    """Simulation goals understood by ``pyslice_suggest_parameters``.

    Each goal changes which physics rules drive the suggested sampling,
    probe design, and frame counts.

    Attributes
    ----------
    DIFFRACTION : str
        Parallel-beam (TEM) diffraction / SAED-like pattern.
    TEM_IMAGING : str
        Parallel-beam real-space imaging (HRTEM-like exit wave).
    HAADF : str
        Convergent-probe STEM imaging with an annular detector.
    FOURDSTEM : str
        Convergent-probe scanning with full diffraction patterns kept.
    TACAW : str
        Vibrational-EELS / phonon spectroscopy from an MD trajectory.

    Methods
    -------
    (inherits enum behavior)

    See Also
    --------
    PySliceService.suggest_parameters : Consumes this goal.
    """

    DIFFRACTION = "diffraction"
    TEM_IMAGING = "tem_imaging"
    HAADF = "haadf"
    FOURDSTEM = "4dstem"
    TACAW = "tacaw"


class StrictModel(BaseModel):
    """Pydantic base model for PySlice MCP tool inputs.

    Applies the ecosystem-standard strictness: whitespace stripping,
    assignment validation, and rejection of unknown fields so typos surface
    as errors instead of being silently ignored.

    Attributes
    ----------
    model_config : pydantic.ConfigDict
        Shared strict configuration.

    Methods
    -------
    (inherits pydantic BaseModel behavior)

    See Also
    --------
    pySEA.sea_eco.mcp.service.StrictModel : The pattern being mirrored.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")


class HandleInput(StrictModel):
    """Input referencing one registered object handle.

    Attributes
    ----------
    handle : str
        Handle returned by a previous PySlice MCP tool.
    response_format : ResponseFormat
        Output format for the response.

    Methods
    -------
    _handle_not_empty(value)
        Field validator rejecting blank handles.
    """

    handle: str = Field(..., description="Object handle returned by a previous pyslice MCP tool.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)

    @field_validator("handle")
    @classmethod
    def _handle_not_empty(cls, value: str) -> str:
        """Validate that a handle string is non-empty.

        Parameters
        ----------
        value : str
            Raw handle string.

        Returns
        -------
        str
            The validated handle.

        Raises
        ------
        ValueError
            If the handle is empty or whitespace.
        """
        if not value.strip():
            raise ValueError("handle cannot be empty")
        return value


class SearchStructuresInput(StrictModel):
    """Input for searching a crystal-structure database.

    Attributes
    ----------
    provider : {"mp", "cod"}
        Database to query (Materials Project or COD).
    formula : str | None
        Chemical formula filter, e.g. ``"SiO2"``.
    elements : list[str] | None
        Element symbols that must all be present.
    limit : int
        Maximum entries to return.
    api_key : str | None
        Materials Project API key; prefer the ``PYSLICE_MP_API_KEY`` env var.
    response_format : ResponseFormat
        Output format for the response.
    """

    provider: Literal["mp", "cod"] = Field(..., description="'mp' = Materials Project (API key), 'cod' = Crystallography Open Database (keyless).")
    formula: Optional[str] = Field(default=None, description="Chemical formula, e.g. 'SiO2' or 'BaTiO3'.")
    elements: Optional[List[str]] = Field(default=None, description="Element symbols that must all be present, e.g. ['Ga', 'N'].")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum entries to return.")
    api_key: Optional[str] = Field(default=None, description="Materials Project API key; falls back to PYSLICE_MP_API_KEY / MP_API_KEY env vars.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class FetchStructureInput(StrictModel):
    """Input for downloading a database entry as a CIF (and optionally loading it).

    Attributes
    ----------
    provider : {"mp", "cod"}
        Database the entry comes from.
    entry_id : str
        Entry id from ``pyslice_search_structures`` (e.g. ``"mp-149"``).
    filename : str | None
        Workspace-relative CIF filename; defaults to ``<provider>_<id>.cif``.
    load : bool
        Also load the CIF into a Trajectory handle.
    timestep_ps : float | None
        Timestep stored on the loaded trajectory (picoseconds).
    api_key : str | None
        Materials Project API key; prefer the env var.
    response_format : ResponseFormat
        Output format for the response.
    """

    provider: Literal["mp", "cod"] = Field(..., description="'mp' or 'cod'.")
    entry_id: str = Field(..., min_length=1, description="Entry id, e.g. 'mp-149' or '1010939'.")
    filename: Optional[str] = Field(default=None, description="Workspace-relative output filename for the CIF.")
    load: bool = Field(default=True, description="Also load the CIF into a Trajectory handle.")
    timestep_ps: Optional[float] = Field(default=None, gt=0, description="Trajectory timestep in ps (single frames ignore it).")
    api_key: Optional[str] = Field(default=None, description="Materials Project API key; falls back to env vars.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class LoadStructureInput(StrictModel):
    """Input for loading a structure or trajectory file.

    Attributes
    ----------
    path : str
        Path to a CIF / XYZ / LAMMPS dump / ASE ``.traj`` file.
    atom_mapping : dict[str, int | str] | None
        LAMMPS type-id → element mapping (keys are stringified ints).
    timestep_ps : float | None
        Timestep in picoseconds for multi-frame files.
    response_format : ResponseFormat
        Output format for the response.
    """

    path: str = Field(..., min_length=1, description="Structure/trajectory file path (CIF via ASE; XYZ/LAMMPS/traj via OVITO).")
    atom_mapping: Optional[Dict[str, int | str]] = Field(
        default=None,
        description="LAMMPS type-id to element mapping, e.g. {'1': 'B', '2': 'N'} or {'1': 5, '2': 7}.",
    )
    timestep_ps: Optional[float] = Field(default=None, gt=0, description="Frame spacing in picoseconds (MD trajectories).")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class TrajectoryOperation(StrictModel):
    """One structure-building operation for ``pyslice_transform_trajectory``.

    Attributes
    ----------
    op : str
        Operation name (see the ``params`` description for each operation's
        expected parameters).
    params : dict
        Operation parameters.
    """

    op: Literal[
        "tile",
        "rotate_to",
        "tilt",
        "slice_positions",
        "slice_timesteps",
        "random_frames",
        "frozen_phonon",
        "fold_to_orthogonal",
        "swap_axes",
    ] = Field(..., description=(
        "tile: {repeats: [nx, ny, nz]} | rotate_to: {direction: [h, k, l]} | "
        "tilt: {alpha_deg, beta_deg} | slice_positions: {x_range?, y_range?, z_range?} (Å pairs) | "
        "slice_timesteps: {i1?, i2?, ith?} | random_frames: {n, seed?} | "
        "frozen_phonon: {n, sigma_A, seed?} | fold_to_orthogonal: {axes?, lengths?} | "
        "swap_axes: {axes: [i, j, k]}"
    ))
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the operation.")


class TransformTrajectoryInput(StrictModel):
    """Input for applying trajectory transforms in sequence.

    Attributes
    ----------
    handle : str
        Trajectory handle to start from (never mutated).
    operations : list[TrajectoryOperation]
        Operations applied left to right; each returns a new trajectory.
    name : str | None
        Optional registry label for the result.
    response_format : ResponseFormat
        Output format for the response.
    """

    handle: str = Field(..., description="Trajectory handle to transform.")
    operations: List[TrajectoryOperation] = Field(..., min_length=1, description="Operations applied in order.")
    name: Optional[str] = Field(default=None, description="Optional registry label for the result.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class SuggestParametersInput(StrictModel):
    """Input for physics-based multislice parameter suggestions.

    Attributes
    ----------
    trajectory_handle : str
        Trajectory the simulation will run on.
    goal : SimulationGoal
        What the simulation is for; selects the applicable rules.
    voltage_eV : float
        Accelerating voltage in eV.
    aperture_mrad : float | None
        Convergence semi-angle; defaults per goal (0 for parallel-beam goals,
        25 mrad for STEM goals).
    max_scattering_angle_mrad : float | None
        Largest scattering angle that must be represented; defaults per goal
        (HAADF uses 1.2 × the ADF outer angle).
    adf_inner_mrad, adf_outer_mrad : float
        ADF detector angles used for the HAADF goal.
    scan_extent_A : float | None
        Length of the square scan region for STEM goals (defaults to the
        structure's lateral extent).
    target_max_frequency_THz : float
        TACAW: highest phonon frequency that must be resolvable.
    target_frequency_resolution_THz : float
        TACAW: desired frequency-bin width.
    response_format : ResponseFormat
        Output format for the response.
    """

    trajectory_handle: str = Field(..., description="Trajectory handle the simulation will run on.")
    goal: SimulationGoal = Field(..., description="Simulation goal driving the parameter rules.")
    voltage_eV: float = Field(default=100e3, gt=0, description="Accelerating voltage in eV.")
    aperture_mrad: Optional[float] = Field(default=None, ge=0, description="Convergence semi-angle in mrad; defaults per goal.")
    max_scattering_angle_mrad: Optional[float] = Field(default=None, gt=0, description="Largest scattering angle to represent, in mrad.")
    adf_inner_mrad: float = Field(default=60.0, gt=0, description="ADF inner collection angle (haadf goal).")
    adf_outer_mrad: float = Field(default=200.0, gt=0, description="ADF outer collection angle (haadf goal).")
    scan_extent_A: Optional[float] = Field(default=None, gt=0, description="Square STEM scan extent in Å; defaults to the cell's lateral extent.")
    target_max_frequency_THz: float = Field(default=30.0, gt=0, description="TACAW: highest frequency to resolve.")
    target_frequency_resolution_THz: float = Field(default=0.3, gt=0, description="TACAW: frequency-bin width.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class RunMDInput(StrictModel):
    """Input for running molecular dynamics with an ML potential.

    Attributes
    ----------
    trajectory_handle : str
        Starting structure (frame 0 is used).
    calculator : {"orb", "fairchem"}
        ML-potential family.
    model_name : str | None
        Model checkpoint; defaults to each calculator's default.
    device : str | None
        'cpu', 'cuda', or 'mps'; auto-selected when omitted.
    temperature_K : float
        Target temperature.
    timestep_fs : float
        MD integration timestep in femtoseconds.
    ensemble : {"nvt", "npt", "nve"}
        Equilibration ensemble.
    production_ensemble : {"nvt", "npt", "nve"} | None
        Production ensemble; 'nve' gives noise-free dynamics for TACAW.
    production_steps : int
        Number of production MD steps.
    save_interval : int
        Save a frame every N steps.
    output_dir : str | None
        Workspace-relative directory for MD outputs.
    response_format : ResponseFormat
        Output format for the response.
    """

    trajectory_handle: str = Field(..., description="Structure handle; frame 0 seeds the MD run.")
    calculator: Literal["orb", "fairchem"] = Field(default="orb", description="ML potential family.")
    model_name: Optional[str] = Field(default=None, description="Model checkpoint name; defaults per calculator.")
    device: Optional[str] = Field(default=None, description="'cpu', 'cuda', or 'mps'.")
    temperature_K: float = Field(default=300.0, gt=0, description="Target temperature in K.")
    timestep_fs: float = Field(default=1.0, gt=0, description="Integration timestep in fs.")
    ensemble: Literal["nvt", "npt", "nve"] = Field(default="nvt", description="Equilibration ensemble.")
    production_ensemble: Optional[Literal["nvt", "npt", "nve"]] = Field(default=None, description="Production ensemble; use 'nve' for TACAW.")
    production_steps: int = Field(default=10000, ge=1, description="Production MD steps.")
    save_interval: int = Field(default=10, ge=1, description="Save every N steps.")
    output_dir: Optional[str] = Field(default=None, description="Workspace-relative output directory.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class SetupMultisliceInput(StrictModel):
    """Input for configuring a multislice calculation.

    Attributes
    ----------
    trajectory_handle : str
        Trajectory to simulate (every frame is propagated).
    aperture_mrad : float
        Convergence semi-angle; 0 = parallel beam (TEM), >0 = STEM probe.
    voltage_eV : float
        Accelerating voltage in eV.
    defocus_A : float
        Probe defocus in Å.
    slice_thickness_A : float
        Multislice slice thickness in Å.
    sampling_A : float
        Real-space sampling in Å/pixel.
    probe_xs, probe_ys : list[float] | None
        Explicit probe grid coordinates in Å.
    probe_grid : dict | None
        Compact grid spec ``{"x": [start, stop, n], "y": [start, stop, n]}``.
    slice_axis : int
        Beam axis (0=x, 1=y, 2=z).
    return_layers : int | str | list[int] | None
        Which wavefunction layers to keep (-1 exit wave, 'all', list, or
        null for HAADF-only runs).
    adf : list[float] | None
        On-the-fly ADF detector ``[inner_mrad, outer_mrad]``.
    aberrations : dict[str, float | list[float]] | None
        abTEM-convention Cnm aberrations, magnitude in Å (value) or
        ``[magnitude_A, angle_rad]`` pairs.
    max_kx, max_ky : float | None
        Optional k-space crops in 1/Å.
    device : str | None
        'cpu', 'cuda', or 'mps'; auto-selected when omitted.
    force_cpu : bool
        Force the numpy/CPU backend.
    cache_wavefunctions : bool
        Write per-frame wavefunction caches under ``psi_data/``.
    name : str | None
        Optional registry label for the calculator.
    response_format : ResponseFormat
        Output format for the response.
    """

    trajectory_handle: str = Field(..., description="Trajectory handle to simulate.")
    aperture_mrad: float = Field(default=0.0, ge=0, description="Convergence semi-angle in mrad; 0 = parallel/TEM.")
    voltage_eV: float = Field(default=100e3, gt=0, description="Accelerating voltage in eV.")
    defocus_A: float = Field(default=0.0, description="Defocus in Å.")
    slice_thickness_A: float = Field(default=0.5, gt=0, description="Slice thickness in Å.")
    sampling_A: float = Field(default=0.1, gt=0, description="Real-space sampling in Å/pixel.")
    probe_xs: Optional[List[float]] = Field(default=None, description="Probe x coordinates in Å.")
    probe_ys: Optional[List[float]] = Field(default=None, description="Probe y coordinates in Å.")
    probe_grid: Optional[Dict[str, List[float]]] = Field(
        default=None, description="Grid spec {'x': [start, stop, n], 'y': [start, stop, n]} in Å; expands to probe_xs/probe_ys."
    )
    slice_axis: int = Field(default=2, ge=0, le=2, description="Beam axis: 0=x, 1=y, 2=z.")
    return_layers: int | str | List[int] | None = Field(
        default=-1, description="-1 exit wave, 'all' every slice, [i, j, ...] selected slices, null = HAADF-only."
    )
    adf: Optional[List[float]] = Field(default=None, min_length=2, max_length=2, description="[inner_mrad, outer_mrad] on-the-fly ADF detector.")
    aberrations: Optional[Dict[str, float | List[float]]] = Field(
        default=None, description="Cnm aberrations (abTEM convention): {'C10': mag_A} or {'C12': [mag_A, angle_rad]}."
    )
    max_kx: Optional[float] = Field(default=None, gt=0, description="Optional kx crop in 1/Å.")
    max_ky: Optional[float] = Field(default=None, gt=0, description="Optional ky crop in 1/Å.")
    device: Optional[str] = Field(default=None, description="'cpu', 'cuda', or 'mps'.")
    force_cpu: bool = Field(default=False, description="Force the numpy/CPU backend.")
    cache_wavefunctions: bool = Field(default=False, description="Write per-frame caches under psi_data/.")
    name: Optional[str] = Field(default=None, description="Optional registry label.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class RunMultisliceInput(StrictModel):
    """Input for executing a configured multislice calculation.

    Attributes
    ----------
    calculator_handle : str
        Handle from ``pyslice_setup_multislice``.
    force_rerun : bool
        Ignore cached frames and recompute.
    response_format : ResponseFormat
        Output format for the response.
    """

    calculator_handle: str = Field(..., description="Calculator handle from pyslice_setup_multislice.")
    force_rerun: bool = Field(default=False, description="Ignore cached frames and recompute.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ComputeHAADFInput(StrictModel):
    """Input for integrating a HAADF/ADF image from wavefunction data.

    Attributes
    ----------
    wf_handle : str
        WFData handle from ``pyslice_run_multislice``.
    inner_mrad, outer_mrad : float
        Annular detector collection angles.
    save_png : str | None
        Workspace-relative PNG path to render the image to.
    response_format : ResponseFormat
        Output format for the response.
    """

    wf_handle: str = Field(..., description="WFData handle.")
    inner_mrad: float = Field(default=60.0, gt=0, description="Inner collection angle in mrad.")
    outer_mrad: float = Field(default=200.0, gt=0, description="Outer collection angle in mrad.")
    save_png: Optional[str] = Field(default=None, description="Workspace-relative PNG output path.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ComputeTACAWInput(StrictModel):
    """Input for converting wavefunction data to TACAW spectral data.

    Attributes
    ----------
    wf_handle : str
        WFData handle from a multi-frame (MD/frozen-phonon) run.
    layer_index : int | None
        Wavefunction layer to transform; defaults to the exit wave.
    temperature_K : float | None
        MD temperature for the Bose detailed-balance correction.
    apply_bose : bool
        Apply the Bose correction (requires ``temperature_K``).
    chunk_size_time : int | None
        Optional FFT chunk length dividing the frame count.
    response_format : ResponseFormat
        Output format for the response.
    """

    wf_handle: str = Field(..., description="WFData handle (multi-frame).")
    layer_index: Optional[int] = Field(default=None, description="Layer to transform; default exit wave.")
    temperature_K: Optional[float] = Field(default=None, gt=0, description="MD temperature for Bose correction.")
    apply_bose: bool = Field(default=False, description="Apply Bose detailed-balance correction.")
    chunk_size_time: Optional[int] = Field(default=None, ge=1, description="FFT chunk length; must divide the frame count.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class TACAWSpectrumInput(StrictModel):
    """Input for extracting a k-integrated TACAW spectrum.

    Attributes
    ----------
    tacaw_handle : str
        TACAWData handle.
    probe_index : int | None
        Probe to extract; mean over all probes when omitted.
    n_peaks : int
        Number of dominant positive-frequency peaks to report.
    response_format : ResponseFormat
        Output format for the response.
    """

    tacaw_handle: str = Field(..., description="TACAWData handle.")
    probe_index: Optional[int] = Field(default=None, ge=0, description="Probe index; mean over probes when omitted.")
    n_peaks: int = Field(default=5, ge=1, le=50, description="Dominant positive-frequency peaks to report.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class SpectrumImageInput(StrictModel):
    """Input for building a real-space map at one phonon frequency.

    Attributes
    ----------
    tacaw_handle : str
        TACAWData handle (needs a probe grid).
    frequency_THz : float
        Frequency at which to map intensity.
    save_png : str | None
        Workspace-relative PNG output path.
    response_format : ResponseFormat
        Output format for the response.
    """

    tacaw_handle: str = Field(..., description="TACAWData handle from a probe-grid run.")
    frequency_THz: float = Field(..., description="Frequency to map, in THz.")
    save_png: Optional[str] = Field(default=None, description="Workspace-relative PNG output path.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class DispersionInput(StrictModel):
    """Input for extracting a phonon dispersion along a k-path.

    Attributes
    ----------
    tacaw_handle : str
        TACAWData handle.
    kx_path, ky_path : list[float] | None
        Explicit k-path coordinates in 1/Å (same length).
    path : dict | None
        Compact spec ``{"from": [kx, ky], "to": [kx, ky], "n": int}``.
    probe_index : int | None
        Probe to extract; mean over all probes when omitted.
    save_png : str | None
        Workspace-relative PNG output path.
    response_format : ResponseFormat
        Output format for the response.
    """

    tacaw_handle: str = Field(..., description="TACAWData handle.")
    kx_path: Optional[List[float]] = Field(default=None, description="kx coordinates of the path in 1/Å.")
    ky_path: Optional[List[float]] = Field(default=None, description="ky coordinates of the path in 1/Å.")
    path: Optional[Dict[str, Any]] = Field(default=None, description="{'from': [kx, ky], 'to': [kx, ky], 'n': int} alternative to explicit paths.")
    probe_index: Optional[int] = Field(default=None, ge=0, description="Probe index; mean over probes when omitted.")
    save_png: Optional[str] = Field(default=None, description="Workspace-relative PNG output path.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class PreviewPotentialInput(StrictModel):
    """Input for rendering the projected potential of a structure.

    Attributes
    ----------
    trajectory_handle : str
        Trajectory handle (frame 0 is used).
    sampling_A : float
        Real-space sampling in Å/pixel.
    slice_thickness_A : float
        Slice thickness in Å.
    slice_axis : int
        Beam axis (0=x, 1=y, 2=z).
    save_png : str | None
        Workspace-relative PNG output path.
    response_format : ResponseFormat
        Output format for the response.
    """

    trajectory_handle: str = Field(..., description="Trajectory handle; frame 0 is rendered.")
    sampling_A: float = Field(default=0.2, gt=0, description="Sampling in Å/pixel.")
    slice_thickness_A: float = Field(default=1.0, gt=0, description="Slice thickness in Å.")
    slice_axis: int = Field(default=2, ge=0, le=2, description="Beam axis: 0=x, 1=y, 2=z.")
    save_png: Optional[str] = Field(default=None, description="Workspace-relative PNG output path.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ExportSeaInput(StrictModel):
    """Input for exporting a result handle to a ``.sea`` file.

    Attributes
    ----------
    handle : str
        WFData/HAADFData/TACAWData handle to export.
    filename : str
        Workspace-relative ``.sea`` output path.
    response_format : ResponseFormat
        Output format for the response.
    """

    handle: str = Field(..., description="WFData/HAADFData/TACAWData handle.")
    filename: str = Field(..., min_length=1, description="Workspace-relative .sea output path.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


def _electron_wavelength_A(voltage_eV: float) -> float:
    """Return the relativistic electron wavelength in Angstroms.

    Uses the standard relativistic de Broglie expression for an electron
    accelerated through ``voltage_eV`` volts.

    Parameters
    ----------
    voltage_eV : float
        Accelerating voltage in eV.

    Returns
    -------
    float
        Wavelength in Å (e.g. ~0.0370 Å at 100 kV).

    Raises
    ------
    ValueError
        If the voltage is not positive.

    See Also
    --------
    PySliceService.suggest_parameters : Main consumer.

    References
    ----------
    .. [1] E. J. Kirkland, "Advanced Computing in Electron Microscopy", Eq. 2.5.
    """
    if voltage_eV <= 0:
        raise ValueError("voltage_eV must be positive")
    return 12.2639 / math.sqrt(voltage_eV + 0.97845e-6 * voltage_eV**2)


class PySliceService:
    """Stateful service used by the PySlice MCP tools.

    Owns the object-handle registry and the workspace directory where CIFs,
    ``.sea`` exports, and PNG previews are written. Every MCP tool is a thin
    wrapper over one method here, so agentic and scripted use share the same
    code path.

    Parameters
    ----------
    workspace : str | pathlib.Path | None, optional
        Directory for generated artifacts; defaults to
        ``PYSLICE_MCP_WORKSPACE`` or the current directory.

    Attributes
    ----------
    workspace : pathlib.Path
        Resolved artifact directory (created on init).
    _objects : dict[str, Any]
        Handle → live-object registry.

    Methods
    -------
    get_conventions()
        Units, workflow, and gotchas — agents call this first.
    search_structures(...), fetch_structure(...), load_structure(...)
        Structure acquisition (databases and files).
    transform_trajectory(...)
        Supercell/orientation/frozen-phonon structure building.
    suggest_parameters(...)
        Physics-grounded multislice parameter advisor.
    run_md(...), setup_multislice(...), run_multislice(...)
        Simulation execution.
    compute_haadf(...), compute_tacaw(...), tacaw_spectrum(...),
    spectrum_image(...), dispersion(...), preview_potential(...)
        Post-processing and previews.
    export_sea(...)
        Persist results as ``.sea`` artifacts.

    Raises
    ------
    (methods raise KeyError/TypeError/ValueError/DatabaseError as documented)

    See Also
    --------
    pySEA.sea_eco.mcp.service.SeaEcoService : The mirrored pattern.

    Notes
    -----
    Handles refer to live in-memory objects; they do not survive a server
    restart. Persist results with :meth:`export_sea`.
    """

    def __init__(self, workspace: str | Path | None = None) -> None:
        """Initialize the registry and workspace directory.

        Parameters
        ----------
        workspace : str | pathlib.Path | None, optional
            Artifact directory; defaults to ``PYSLICE_MCP_WORKSPACE`` or the
            current directory.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If the workspace directory cannot be created.
        """
        workspace = workspace or os.environ.get("PYSLICE_MCP_WORKSPACE", ".")
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._objects: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Conventions / registry
    # ------------------------------------------------------------------

    def get_conventions(self) -> Dict[str, Any]:
        """Return MCP-facing conventions, units, and workflow guidance.

        Agents should call this first in zero-context sessions: it documents
        the unit system, the canonical workflow, PySlice's gotchas, and one
        example call per major tool.

        Returns
        -------
        dict
            Conventions payload (tool prefix, units, workflow, gotchas,
            examples).

        See Also
        --------
        suggest_parameters : Physics rules referenced by the workflow.
        """
        return {
            "tool_prefix": "pyslice_",
            "call_first": "pyslice_get_conventions",
            "units": {
                "length": "Angstrom (Å) everywhere: sampling, defocus, slice thickness, probe positions, sigma.",
                "angles": "mrad for aperture and detector angles; degrees in transform 'tilt'; radians only inside aberration [mag, angle] pairs.",
                "voltage": "eV (e.g. 100e3 = 100 kV).",
                "trajectory_timestep": "picoseconds between frames.",
                "md_timestep": "femtoseconds (MD integration step) — note the ps/fs difference.",
                "frequency": "THz for TACAW spectra.",
                "k_space": "1/Å (not mrad) for kx/ky arrays.",
            },
            "workflow": [
                "1. structure: pyslice_search_structures + pyslice_fetch_structure (or pyslice_load_structure for local files)",
                "2. build: pyslice_transform_trajectory (tile / rotate_to zone axis / frozen_phonon ...)",
                "3. advise: pyslice_suggest_parameters (goal-specific sampling, slices, probes, frames)",
                "4. (optional) pyslice_run_md for thermal trajectories (TACAW needs this or frozen_phonon)",
                "5. pyslice_setup_multislice -> check grid/memory in the response -> pyslice_run_multislice",
                "6. post-process: pyslice_compute_haadf / pyslice_compute_tacaw / pyslice_tacaw_spectrum / pyslice_spectrum_image / pyslice_dispersion",
                "7. persist: pyslice_export_sea (.sea files open across the pySEA ecosystem)",
            ],
            "gotchas": {
                "aperture": "aperture_mrad=0 means parallel beam (TEM/diffraction); >0 means a convergent STEM probe.",
                "atom_types": "Element symbols (strings) on the ASE/CIF path, integer type ids on the OVITO path — pass atom_mapping for LAMMPS files.",
                "supercell": "Tiling, orientation, tilts, and frozen-phonon displacement are trajectory transforms, not setup() arguments.",
                "frames": "Every trajectory frame is propagated; frame count = frozen-phonon configs or MD snapshots.",
                "handles": "Handles are live in-memory objects; they do not survive a server restart — export .sea files to persist.",
                "blocking": "pyslice_run_multislice and pyslice_run_md block until finished; check the estimated memory/grid in setup output first.",
                "mp_api_key": "Materials Project needs a free API key (PYSLICE_MP_API_KEY or MP_API_KEY env var); COD is keyless.",
            },
            "examples": {
                "fetch": {"tool": "pyslice_fetch_structure", "params": {"provider": "cod", "entry_id": "1010939", "load": True}},
                "advise": {"tool": "pyslice_suggest_parameters", "params": {"trajectory_handle": "Trajectory:cod_1010939", "goal": "haadf"}},
                "setup": {
                    "tool": "pyslice_setup_multislice",
                    "params": {
                        "trajectory_handle": "Trajectory:cod_1010939",
                        "aperture_mrad": 25,
                        "voltage_eV": 100e3,
                        "sampling_A": 0.08,
                        "adf": [60, 200],
                        "probe_grid": {"x": [0, 10, 20], "y": [0, 10, 20]},
                    },
                },
            },
        }

    def get_workspace(self) -> Dict[str, Any]:
        """Return workspace and registry metadata.

        Returns
        -------
        dict
            Workspace path and handle count.
        """
        return {"workspace": str(self.workspace), "handle_count": len(self._objects)}

    def list_handles(self) -> Dict[str, Any]:
        """Return a summary of every registered object handle.

        Returns
        -------
        dict
            ``{"handles": [{"handle": ..., **summary}, ...]}`` sorted by
            handle name.
        """
        return {"handles": [{"handle": h, **self._summary(obj)} for h, obj in sorted(self._objects.items())]}

    def describe_handle(self, handle: str) -> Dict[str, Any]:
        """Describe one registered object.

        Parameters
        ----------
        handle : str
            Registered handle.

        Returns
        -------
        dict
            Type-specific summary of the object.

        Raises
        ------
        KeyError
            If the handle is unknown.
        """
        return {"handle": handle, **self._summary(self._get(handle))}

    # ------------------------------------------------------------------
    # Structure acquisition
    # ------------------------------------------------------------------

    def search_structures(
        self,
        provider: str,
        formula: Optional[str],
        elements: Optional[Sequence[str]],
        limit: int,
        api_key: Optional[str],
    ) -> Dict[str, Any]:
        """Search Materials Project or COD for structures.

        Parameters
        ----------
        provider : {"mp", "cod"}
            Database to query.
        formula : str | None
            Chemical formula filter.
        elements : Sequence[str] | None
            Elements that must all be present.
        limit : int
            Maximum entries to return.
        api_key : str | None
            Materials Project API key (env-var fallback).

        Returns
        -------
        dict
            ``{"provider", "count", "entries": [...]}`` where each entry's
            ``id`` feeds :meth:`fetch_structure`.

        Raises
        ------
        ValueError
            If neither formula nor elements is given.
        pyslice.io.databases.DatabaseError
            On network/authentication failure.
        """
        from ..io.databases import search_structures as _search

        entries = _search(provider, formula=formula, elements=elements, limit=limit, api_key=api_key)
        return {"provider": provider, "count": len(entries), "entries": entries}

    def fetch_structure(
        self,
        provider: str,
        entry_id: str,
        filename: Optional[str],
        load: bool,
        timestep_ps: Optional[float],
        api_key: Optional[str],
    ) -> Dict[str, Any]:
        """Download a database entry as a CIF and optionally load it.

        Parameters
        ----------
        provider : {"mp", "cod"}
            Database the entry comes from.
        entry_id : str
            Entry id from :meth:`search_structures`.
        filename : str | None
            Workspace-relative CIF filename.
        load : bool
            Also load into a Trajectory handle.
        timestep_ps : float | None
            Timestep for the loaded trajectory.
        api_key : str | None
            Materials Project API key (env-var fallback).

        Returns
        -------
        dict
            CIF path plus, when ``load`` is true, the new trajectory handle
            and its summary.

        Raises
        ------
        pyslice.io.databases.DatabaseError
            If the entry cannot be retrieved.
        """
        from ..io.databases import fetch_cif
        from ..io.loader import Loader

        target_dir = self.workspace / "structures"
        cif_path = fetch_cif(provider, entry_id, output_dir=target_dir, filename=filename, api_key=api_key)
        payload: Dict[str, Any] = {"provider": provider, "entry_id": entry_id, "cif_path": str(cif_path)}
        if load:
            trajectory = Loader(filename=str(cif_path), timestep=timestep_ps).load()
            payload.update(self._register(trajectory, preferred=cif_path.stem))
        return payload

    def load_structure(
        self,
        path: str,
        atom_mapping: Optional[Dict[str, int | str]],
        timestep_ps: Optional[float],
    ) -> Dict[str, Any]:
        """Load a structure/trajectory file into a Trajectory handle.

        Parameters
        ----------
        path : str
            CIF / XYZ / LAMMPS dump / ASE ``.traj`` file path.
        atom_mapping : dict[str, int | str] | None
            LAMMPS type-id → element mapping (string keys are coerced to
            int).
        timestep_ps : float | None
            Frame spacing in picoseconds.

        Returns
        -------
        dict
            New trajectory handle and summary.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        ValueError
            If the atom mapping keys are not integers.
        """
        from ..io.loader import Loader

        mapping: Optional[Dict[int, int | str]] = None
        if atom_mapping:
            try:
                mapping = {int(k): v for k, v in atom_mapping.items()}
            except ValueError as exc:
                raise ValueError("atom_mapping keys must be integer type ids, e.g. {'1': 'B'}") from exc
        trajectory = Loader(filename=path, atom_mapping=mapping, timestep=timestep_ps).load()
        return self._register(trajectory, preferred=Path(path).stem)

    def transform_trajectory(
        self,
        handle: str,
        operations: Sequence[TrajectoryOperation],
        name: Optional[str],
    ) -> Dict[str, Any]:
        """Apply structure-building transforms to a trajectory.

        Operations run left to right, each producing a new trajectory; the
        source trajectory is never mutated.

        Parameters
        ----------
        handle : str
            Trajectory handle to start from.
        operations : Sequence[TrajectoryOperation]
            Operations with their parameters (see the input model).
        name : str | None
            Optional registry label for the result.

        Returns
        -------
        dict
            New trajectory handle, summary, and the applied operation list.

        Raises
        ------
        KeyError
            If the handle is unknown.
        TypeError
            If the handle is not a Trajectory.
        ValueError
            If an operation's parameters are invalid.
        """
        from ..multislice.trajectory import Trajectory

        trajectory = self._require_type(handle, Trajectory)
        applied: List[str] = []
        for operation in operations:
            params = dict(operation.params)
            op = operation.op
            if op == "tile":
                trajectory = trajectory.tile_positions(tuple(int(v) for v in params["repeats"]))
            elif op == "rotate_to":
                trajectory = trajectory.rotate_to(tuple(params["direction"]))
            elif op == "tilt":
                trajectory = trajectory.tilt_positions(
                    alpha=math.radians(float(params.get("alpha_deg", 0.0))),
                    beta=math.radians(float(params.get("beta_deg", 0.0))),
                )
            elif op == "slice_positions":
                trajectory = trajectory.slice_positions(
                    x_range=tuple(params["x_range"]) if params.get("x_range") else None,
                    y_range=tuple(params["y_range"]) if params.get("y_range") else None,
                    z_range=tuple(params["z_range"]) if params.get("z_range") else None,
                )
            elif op == "slice_timesteps":
                trajectory = trajectory.slice_timesteps(
                    i1=int(params.get("i1", 0)),
                    i2=int(params["i2"]) if params.get("i2") is not None else None,
                    ith=int(params.get("ith", 1)),
                )
            elif op == "random_frames":
                trajectory = trajectory.random_frames(int(params["n"]), seed=params.get("seed"))
            elif op == "frozen_phonon":
                trajectory = trajectory.generate_random_displacements(
                    int(params["n"]), float(params["sigma_A"]), seed=params.get("seed")
                )
            elif op == "fold_to_orthogonal":
                trajectory = trajectory.fold_positions_to_orthogonal_box(
                    axes=tuple(params.get("axes", (0, 1, 2))),
                    lengths=tuple(params["lengths"]) if params.get("lengths") else None,
                )
            elif op == "swap_axes":
                trajectory = trajectory.swap_axes(list(params["axes"]))
            applied.append(op)
        result = self._register(trajectory, preferred=name or f"{handle.split(':', 1)[-1]}-built")
        result["applied_operations"] = applied
        return result

    # ------------------------------------------------------------------
    # Parameter advisor
    # ------------------------------------------------------------------

    def suggest_parameters(self, params: SuggestParametersInput) -> Dict[str, Any]:
        """Suggest physics-grounded multislice parameters for a goal.

        Encodes the parameter-selection rules from the
        ``simulation-parameter-selection`` skill: antialiasing-limited
        sampling, evenly dividing slice thickness, k-space resolution and
        probe-wraparound driven tiling, Nyquist probe steps, and MD/frozen-
        phonon frame counts for the requested frequency window.

        Parameters
        ----------
        params : SuggestParametersInput
            Validated goal, trajectory handle, and optics.

        Returns
        -------
        dict
            ``suggested`` kwargs for :meth:`setup_multislice` (plus
            trajectory-transform and MD advice where applicable) and a
            ``justification`` entry per suggestion.

        Raises
        ------
        KeyError
            If the trajectory handle is unknown.
        TypeError
            If the handle is not a Trajectory.

        See Also
        --------
        setup_multislice : Consumes the suggested values.

        Notes
        -----
        Rules of thumb, not guarantees: convergence should still be checked
        by refining sampling/slice thickness until the observable stops
        changing.
        """
        from ..multislice.trajectory import Trajectory

        trajectory = self._require_type(params.trajectory_handle, Trajectory)
        goal = params.goal
        wavelength = _electron_wavelength_A(params.voltage_eV)
        extent = trajectory.extent
        lateral = float(min(extent[0], extent[1]))
        beam_height = float(extent[2])

        stem_goal = goal in (SimulationGoal.HAADF, SimulationGoal.FOURDSTEM)
        aperture = params.aperture_mrad
        if aperture is None:
            aperture = 25.0 if stem_goal else 0.0

        if params.max_scattering_angle_mrad is not None:
            theta_max_mrad = params.max_scattering_angle_mrad
        elif goal == SimulationGoal.HAADF:
            theta_max_mrad = 1.2 * params.adf_outer_mrad
        elif stem_goal:
            theta_max_mrad = max(3.0 * aperture, 60.0)
        else:
            theta_max_mrad = 50.0

        # Antialiasing band limit: usable k_max = 1/(3*sampling) -> sampling = lambda/(3*theta)
        sampling = wavelength / (3.0 * theta_max_mrad * 1e-3)

        # Slice thickness: target 0.5 A but divide the cell height evenly.
        target_slice = 0.5
        n_slices = max(1, round(beam_height / target_slice)) if beam_height > 0 else 1
        slice_thickness = beam_height / n_slices if beam_height > 0 else target_slice

        suggested: Dict[str, Any] = {
            "voltage_eV": params.voltage_eV,
            "aperture_mrad": aperture,
            "sampling_A": round(sampling, 4),
            "slice_thickness_A": round(slice_thickness, 4),
        }
        justification: Dict[str, str] = {
            "sampling_A": (
                f"lambda={wavelength:.5f} Å at {params.voltage_eV / 1e3:.0f} kV; the multislice band limit keeps "
                f"k <= 1/(3*sampling), so representing {theta_max_mrad:.0f} mrad needs sampling <= "
                f"lambda/(3*theta) = {sampling:.4f} Å/px."
            ),
            "slice_thickness_A": (
                f"cell height along the beam is {beam_height:.2f} Å; {n_slices} slices of "
                f"{slice_thickness:.3f} Å divide it evenly (target ~0.5 Å)."
            ),
            "aperture_mrad": (
                "0 mrad = parallel beam for TEM/diffraction goals." if aperture == 0
                else f"{aperture:.0f} mrad convergent probe (typical aberration-corrected STEM range 20-35 mrad)."
            ),
        }

        # Lateral extent requirements -> tiling advice.
        min_extent = 0.0
        reasons: List[str] = []
        if aperture > 0:
            probe_diameter = 1.22 * wavelength / (aperture * 1e-3)
            min_extent = max(min_extent, 4.0 * probe_diameter)
            reasons.append(f"probe diameter ~{probe_diameter:.2f} Å (1.22*lambda/alpha) needs >=4x cell width to avoid wraparound")
            dk_needed = (aperture * 1e-3) / wavelength / 5.0
            min_extent = max(min_extent, 1.0 / dk_needed)
            reasons.append("k-space step dk=1/L should sample the aperture disk with >=5 points")
        if goal == SimulationGoal.TACAW:
            min_extent = max(min_extent, 20.0)
            reasons.append("phonon dispersion needs enough unit cells for k-resolution (dk = 1/L); >=~20 Å laterally as a floor")
        if min_extent > 0 and lateral > 0 and lateral < min_extent:
            repeat = int(math.ceil(min_extent / lateral))
            suggested["tile_repeats"] = [repeat, repeat, 1]
            justification["tile_repeats"] = (
                f"cell is {lateral:.2f} Å wide but {min_extent:.1f} Å is needed ({'; '.join(reasons)}); "
                f"apply pyslice_transform_trajectory op 'tile' with repeats [{repeat}, {repeat}, 1]."
            )

        # Probe grid for scanning goals.
        if stem_goal:
            probe_step = wavelength / (4.0 * aperture * 1e-3)
            scan = params.scan_extent_A or lateral
            n_probe = max(2, int(math.ceil(scan / probe_step)))
            suggested["probe_step_A"] = round(probe_step, 4)
            suggested["probe_grid"] = {"x": [0.0, round(scan, 2), n_probe], "y": [0.0, round(scan, 2), n_probe]}
            justification["probe_grid"] = (
                f"image Nyquist: probe step <= lambda/(4*alpha) = {probe_step:.3f} Å; scanning {scan:.1f} Å "
                f"needs a {n_probe}x{n_probe} grid."
            )
        if goal == SimulationGoal.HAADF:
            suggested["adf"] = [params.adf_inner_mrad, params.adf_outer_mrad]
            suggested["return_layers"] = None
            justification["adf"] = (
                f"[{params.adf_inner_mrad:.0f}, {params.adf_outer_mrad:.0f}] mrad annulus; return_layers=null "
                "accumulates ADF on the fly without storing wavefunctions."
            )

        # Frame counts: frozen phonon vs MD sampling.
        if goal == SimulationGoal.TACAW:
            dt_frame = 1.0 / (2.0 * params.target_max_frequency_THz)  # ps
            n_frames = int(math.ceil(1.0 / (params.target_frequency_resolution_THz * dt_frame)))
            suggested["md_plan"] = {
                "frame_spacing_ps": round(dt_frame, 5),
                "n_frames": n_frames,
                "total_time_ps": round(dt_frame * n_frames, 3),
                "production_ensemble": "nve",
            }
            justification["md_plan"] = (
                f"time Nyquist: frame spacing <= 1/(2*f_max) = {dt_frame * 1e3:.1f} fs for f_max="
                f"{params.target_max_frequency_THz:.0f} THz; resolution df=1/T_total -> "
                f"{n_frames} frames ({dt_frame * n_frames:.1f} ps). Run production in NVE for noise-free dynamics."
            )
        elif goal in (SimulationGoal.HAADF, SimulationGoal.FOURDSTEM):
            suggested["frozen_phonon_frames"] = 12
            justification["frozen_phonon_frames"] = (
                "thermal diffuse scattering converges around 8-16 frozen-phonon configurations; use "
                "pyslice_transform_trajectory op 'frozen_phonon' with n=12 and sigma_A~0.05-0.1."
            )
        else:
            suggested["frozen_phonon_frames"] = 1
            justification["frozen_phonon_frames"] = "a static pattern needs a single frame; add frozen-phonon frames for TDS realism."

        # Memory estimate for the wavefunction array (complex64).
        grid_l = max(lateral, min_extent) if min_extent else lateral
        nx = int(grid_l / sampling) + 1 if grid_l > 0 else 0
        n_probes = suggested.get("probe_grid", {"x": [0, 0, 1]})["x"][2] ** 2 if stem_goal else 1
        n_frames_est = suggested.get("md_plan", {}).get("n_frames", suggested.get("frozen_phonon_frames", 1))
        est_bytes = n_probes * n_frames_est * nx * nx * 8
        suggested["estimated_wavefunction_GiB"] = round(est_bytes / 1024**3, 3)
        justification["estimated_wavefunction_GiB"] = (
            f"complex64 * {n_probes} probes * {n_frames_est} frames * {nx}^2 pixels (exit wave only); "
            "reduce with max_kx/max_ky crops, return_layers=null (HAADF), or coarser sampling."
        )

        return {
            "goal": goal.value,
            "wavelength_A": round(wavelength, 5),
            "structure": {
                "n_atoms": trajectory.n_atoms,
                "n_frames": trajectory.n_frames,
                "extent_A": [round(float(v), 3) for v in extent],
            },
            "suggested": suggested,
            "justification": justification,
        }

    # ------------------------------------------------------------------
    # Simulation execution
    # ------------------------------------------------------------------

    def run_md(self, params: RunMDInput) -> Dict[str, Any]:
        """Run ML-potential molecular dynamics from a structure handle.

        Converts frame 0 of the trajectory to ASE atoms, runs equilibration
        plus production MD with the chosen ML calculator, and registers the
        resulting multi-frame trajectory.

        Parameters
        ----------
        params : RunMDInput
            Validated MD settings.

        Returns
        -------
        dict
            New trajectory handle and summary.

        Raises
        ------
        KeyError
            If the trajectory handle is unknown.
        RuntimeError
            If the ML-potential stack is unavailable; the message names the
            install command.

        Notes
        -----
        Blocking and potentially long; ML weights are downloaded on first
        use. Requires the ``[md]`` extra (Python 3.12).
        """
        from ..multislice.trajectory import Trajectory

        trajectory = self._require_type(params.trajectory_handle, Trajectory)
        atoms = self._trajectory_to_ase(trajectory)

        try:
            from ..md.molecular_dynamics import FAIRChemMDCalculator, ORBMDCalculator
        except Exception as exc:
            raise RuntimeError(
                f"MD stack unavailable ({exc}). Install the extra: pip install 'pyslice[md]' (Python 3.12)."
            ) from exc

        kwargs: Dict[str, Any] = {"device": params.device or "cpu"}
        if params.model_name:
            kwargs["model_name"] = params.model_name
        calculator = ORBMDCalculator(**kwargs) if params.calculator == "orb" else FAIRChemMDCalculator(**kwargs)

        output_dir = self._workspace_path(params.output_dir) if params.output_dir else self.workspace / "md"
        output_dir.mkdir(parents=True, exist_ok=True)
        calculator.setup(
            atoms,
            temperature=params.temperature_K,
            timestep=params.timestep_fs,
            ensemble=params.ensemble,
            production_ensemble=params.production_ensemble,
            production_steps=params.production_steps,
            save_interval=params.save_interval,
            output_dir=output_dir,
        )
        result = calculator.run()
        payload = self._register(result, preferred=f"md-{params.calculator}")
        payload["output_dir"] = str(output_dir)
        return payload

    def setup_multislice(self, params: SetupMultisliceInput) -> Dict[str, Any]:
        """Configure a multislice calculation and register the calculator.

        Runs ``MultisliceCalculator.setup`` with the mapped parameters,
        applies aberrations to the probe, and reports the resulting grid and
        a wavefunction-memory estimate so agents can sanity-check before the
        blocking run.

        Parameters
        ----------
        params : SetupMultisliceInput
            Validated multislice settings.

        Returns
        -------
        dict
            Calculator handle, grid summary (nx/ny/nz, probes, frames), and
            estimated wavefunction memory in GiB.

        Raises
        ------
        KeyError
            If the trajectory handle is unknown.
        TypeError
            If the handle is not a Trajectory.
        ValueError
            If the probe grid spec is malformed.
        """
        from ..multislice.calculators import MultisliceCalculator
        from ..multislice.trajectory import Trajectory

        trajectory = self._require_type(params.trajectory_handle, Trajectory)

        probe_xs, probe_ys = params.probe_xs, params.probe_ys
        if params.probe_grid is not None:
            grid = params.probe_grid
            if set(grid) != {"x", "y"} or any(len(grid[k]) != 3 for k in ("x", "y")):
                raise ValueError("probe_grid must be {'x': [start, stop, n], 'y': [start, stop, n]}")
            probe_xs = np.linspace(grid["x"][0], grid["x"][1], int(grid["x"][2])).tolist()
            probe_ys = np.linspace(grid["y"][0], grid["y"][1], int(grid["y"][2])).tolist()

        calculator = MultisliceCalculator(device=params.device, force_cpu=params.force_cpu)
        setup_kwargs: Dict[str, Any] = {
            "aperture": params.aperture_mrad,
            "voltage_eV": params.voltage_eV,
            "defocus": params.defocus_A,
            "slice_thickness": params.slice_thickness_A,
            "sampling": params.sampling_A,
            "probe_xs": probe_xs,
            "probe_ys": probe_ys,
            "slice_axis": params.slice_axis,
            "return_layers": params.return_layers,
            "cache_wavefunctions": params.cache_wavefunctions,
        }
        if params.adf is not None:
            setup_kwargs["ADF"] = tuple(params.adf)
        if params.max_kx is not None:
            setup_kwargs["max_kx"] = params.max_kx
        if params.max_ky is not None:
            setup_kwargs["max_ky"] = params.max_ky
        calculator.setup(trajectory, **setup_kwargs)

        if params.aberrations:
            aberrations = {k: tuple(v) if isinstance(v, list) else v for k, v in params.aberrations.items()}
            calculator.base_probe.aberrate(aberrations)

        result = self._register(calculator, preferred=params.name or "multislice")
        n_layers = len(calculator._return_layers) or 1
        n_copies = calculator.base_probe._array.shape[0]
        n_probes = n_copies * len(calculator.probe_positions)
        est_bytes = n_probes * trajectory.n_frames * calculator.nx * calculator.ny * n_layers * 8
        result.update({
            "grid": {"nx": calculator.nx, "ny": calculator.ny, "n_slices": calculator.nz},
            "n_probes": n_probes,
            "n_frames": trajectory.n_frames,
            "returned_layers": len(calculator._return_layers),
            "estimated_wavefunction_GiB": round(est_bytes / 1024**3, 3),
            "device": str(calculator.device),
            "cache_dir": str(calculator.output_dir),
        })
        return result

    def run_multislice(self, calculator_handle: str, force_rerun: bool) -> Dict[str, Any]:
        """Execute a configured multislice calculation.

        Blocking; propagates every trajectory frame. Returns a WFData handle
        and, when the calculator was set up with an ADF detector, a
        HAADFData handle accumulated on the fly.

        Parameters
        ----------
        calculator_handle : str
            Handle from :meth:`setup_multislice`.
        force_rerun : bool
            Ignore cached frames and recompute.

        Returns
        -------
        dict
            ``wf`` (WFData handle + summary) and optionally ``haadf``.

        Raises
        ------
        KeyError
            If the handle is unknown.
        TypeError
            If the handle is not a MultisliceCalculator.
        """
        from ..multislice.calculators import MultisliceCalculator

        calculator = self._require_type(calculator_handle, MultisliceCalculator)
        output = calculator.run(force_rerun=force_rerun)
        payload: Dict[str, Any] = {}
        if isinstance(output, tuple):
            wf_data, haadf = output
            payload["wf"] = self._register(wf_data, preferred="wf")
            payload["haadf"] = self._register(haadf, preferred="haadf")
        else:
            payload["wf"] = self._register(output, preferred="wf")
        return payload

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def compute_haadf(self, params: ComputeHAADFInput) -> Dict[str, Any]:
        """Integrate an annular dark-field image from wavefunction data.

        Parameters
        ----------
        params : ComputeHAADFInput
            WFData handle, detector angles, optional PNG path.

        Returns
        -------
        dict
            HAADFData handle, image shape, and PNG path when rendered.

        Raises
        ------
        KeyError
            If the handle is unknown.
        TypeError
            If the handle is not WFData.
        ValueError
            If ``outer_mrad <= inner_mrad``.
        """
        from ..postprocessing.haadf_data import HAADFData
        from ..postprocessing.wf_data import WFData

        if params.outer_mrad <= params.inner_mrad:
            raise ValueError("outer_mrad must exceed inner_mrad")
        wf_data = self._require_type(params.wf_handle, WFData)
        haadf = HAADFData(wf_data)
        haadf.calculateADF(params.inner_mrad, params.outer_mrad)
        payload = self._register(haadf, preferred=f"haadf-{params.inner_mrad:.0f}-{params.outer_mrad:.0f}")
        payload["image_shape"] = list(np.shape(haadf.data))
        if params.save_png:
            png_path = self._workspace_path(params.save_png)
            haadf.plot(filename=str(png_path))
            payload["png_path"] = str(png_path)
        return payload

    def compute_tacaw(self, params: ComputeTACAWInput) -> Dict[str, Any]:
        """Convert multi-frame wavefunction data to TACAW spectral data.

        Parameters
        ----------
        params : ComputeTACAWInput
            WFData handle and spectral options.

        Returns
        -------
        dict
            TACAWData handle, frequency range and step in THz.

        Raises
        ------
        KeyError
            If the handle is unknown.
        TypeError
            If the handle is not WFData.
        ValueError
            If the run has fewer than two frames (no time axis), or Bose
            correction is requested without a temperature.
        """
        from ..postprocessing.tacaw_data import TACAWData
        from ..postprocessing.wf_data import WFData

        wf_data = self._require_type(params.wf_handle, WFData)
        if wf_data.time is None or len(wf_data.time) < 2:
            raise ValueError(
                "TACAW needs a multi-frame run (MD or frozen-phonon trajectory); this WFData has a single frame."
            )
        tacaw = TACAWData(
            wf_data,
            layer_index=params.layer_index,
            temperature_K=params.temperature_K,
            apply_bose=params.apply_bose,
            chunk_size_time=params.chunk_size_time,
        )
        payload = self._register(tacaw, preferred="tacaw")
        frequencies = tacaw.frequencies
        payload.update({
            "frequency_range_THz": [float(frequencies.min()), float(frequencies.max())],
            "frequency_step_THz": float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.0,
        })
        return payload

    def tacaw_spectrum(self, params: TACAWSpectrumInput) -> Dict[str, Any]:
        """Extract a k-integrated spectrum and its dominant peaks.

        Parameters
        ----------
        params : TACAWSpectrumInput
            TACAWData handle, optional probe index, peak count.

        Returns
        -------
        dict
            Spectrum array handle plus the top positive-frequency peaks as
            ``[{"frequency_THz", "intensity"}, ...]``.

        Raises
        ------
        KeyError
            If the handle is unknown.
        TypeError
            If the handle is not TACAWData.
        """
        from ..postprocessing.tacaw_data import TACAWData

        tacaw = self._require_type(params.tacaw_handle, TACAWData)
        spectrum = np.asarray(tacaw.spectrum(probe_index=params.probe_index))
        frequencies = tacaw.frequencies
        positive = frequencies > 0
        order = np.argsort(spectrum[positive])[::-1][: params.n_peaks]
        peak_freqs = frequencies[positive][order]
        peak_vals = spectrum[positive][order]
        payload = self._register(spectrum, preferred="tacaw-spectrum")
        payload["peaks"] = [
            {"frequency_THz": float(f), "intensity": float(v)} for f, v in zip(peak_freqs, peak_vals)
        ]
        return payload

    def spectrum_image(self, params: SpectrumImageInput) -> Dict[str, Any]:
        """Map TACAW intensity at one frequency over the probe grid.

        Parameters
        ----------
        params : SpectrumImageInput
            TACAWData handle, frequency, optional PNG path.

        Returns
        -------
        dict
            Image array handle (probe grid shape when the probes form a
            grid), the actual frequency used, and PNG path when rendered.

        Raises
        ------
        KeyError
            If the handle is unknown.
        TypeError
            If the handle is not TACAWData.
        """
        from ..postprocessing.tacaw_data import TACAWData

        tacaw = self._require_type(params.tacaw_handle, TACAWData)
        values = np.asarray(tacaw.spectrum_image(params.frequency_THz))
        frequencies = tacaw.frequencies
        actual = float(frequencies[int(np.argmin(np.abs(frequencies - params.frequency_THz)))])

        xs = np.unique([p[0] for p in tacaw.probe_positions])
        ys = np.unique([p[1] for p in tacaw.probe_positions])
        if len(xs) * len(ys) == values.size:
            values = values.reshape(len(ys), len(xs))  # probe loop order: y fast inside meshgrid rows
        payload = self._register(values, preferred=f"spectrum-image-{actual:.2f}THz")
        payload["frequency_THz"] = actual
        if params.save_png:
            png_path = self._workspace_path(params.save_png)
            self._save_heatmap(values, str(png_path), title=f"TACAW {actual:.2f} THz", xlabel="probe x", ylabel="probe y")
            payload["png_path"] = str(png_path)
        return payload

    def dispersion(self, params: DispersionInput) -> Dict[str, Any]:
        """Extract a phonon dispersion along a k-path.

        Parameters
        ----------
        params : DispersionInput
            TACAWData handle plus an explicit k-path or a ``path`` spec.

        Returns
        -------
        dict
            Dispersion array handle ``(n_frequencies, n_k)``, the k-path
            used, and PNG path when rendered.

        Raises
        ------
        KeyError
            If the handle is unknown.
        TypeError
            If the handle is not TACAWData.
        ValueError
            If no k-path is given or lengths mismatch.
        """
        from ..postprocessing.tacaw_data import TACAWData

        tacaw = self._require_type(params.tacaw_handle, TACAWData)
        if params.path is not None:
            start = params.path.get("from")
            stop = params.path.get("to")
            n = int(params.path.get("n", 50))
            if start is None or stop is None:
                raise ValueError("path must be {'from': [kx, ky], 'to': [kx, ky], 'n': int}")
            kx_path = np.linspace(start[0], stop[0], n)
            ky_path = np.linspace(start[1], stop[1], n)
        elif params.kx_path is not None and params.ky_path is not None:
            if len(params.kx_path) != len(params.ky_path):
                raise ValueError("kx_path and ky_path must have the same length")
            kx_path = np.asarray(params.kx_path)
            ky_path = np.asarray(params.ky_path)
        else:
            raise ValueError("Provide either kx_path+ky_path or a path spec {'from': ..., 'to': ..., 'n': ...}")

        result = tacaw.dispersion(kx_path, ky_path, probe_index=params.probe_index)
        payload = self._register(np.asarray(result), preferred="dispersion")
        payload["k_path"] = {"kx": [float(v) for v in kx_path[:3]] + (["..."] if len(kx_path) > 3 else []),
                             "n_points": int(len(kx_path))}
        payload["frequency_range_THz"] = [float(tacaw.frequencies.min()), float(tacaw.frequencies.max())]
        if params.save_png:
            png_path = self._workspace_path(params.save_png)
            self._save_heatmap(np.asarray(result), str(png_path), title="Dispersion", xlabel="k-path index", ylabel="frequency index")
            payload["png_path"] = str(png_path)
        return payload

    def preview_potential(self, params: PreviewPotentialInput) -> Dict[str, Any]:
        """Render the projected potential of a structure as a sanity check.

        Builds the frame-0 Kirkland potential on the requested grid, flattens
        it along the beam axis, and registers the projected array.

        Parameters
        ----------
        params : PreviewPotentialInput
            Trajectory handle, grid settings, optional PNG path.

        Returns
        -------
        dict
            Projected-potential array handle, grid shape, and PNG path when
            rendered.

        Raises
        ------
        KeyError
            If the handle is unknown.
        TypeError
            If the handle is not a Trajectory.
        """
        from ..backend import make_backend, to_numpy
        from ..multislice.potentials import Potential, grid_from_trajectory
        from ..multislice.trajectory import Trajectory

        trajectory = self._require_type(params.trajectory_handle, Trajectory)
        backend = make_backend("cpu")
        xs, ys, zs, lx, ly, lz = grid_from_trajectory(
            trajectory, sampling=params.sampling_A, slice_thickness=params.slice_thickness_A
        )
        symbols = self._atom_symbols(trajectory)
        potential = Potential(
            xs, ys, zs, trajectory.positions[0], symbols,
            backend=backend, kind="kirkland", slice_axis=params.slice_axis,
        )
        potential.build()
        potential.flatten()
        projected = np.absolute(to_numpy(potential.array))[:, :, 0]
        payload = self._register(projected, preferred="potential-preview")
        payload["grid"] = {"nx": len(xs), "ny": len(ys), "n_slices": len(zs)}
        payload["box_A"] = [float(lx), float(ly), float(lz)]
        if params.save_png:
            png_path = self._workspace_path(params.save_png)
            self._save_heatmap(projected.T[::-1, :], str(png_path), title="Projected potential", xlabel="x (Å)", ylabel="y (Å)")
            payload["png_path"] = str(png_path)
        return payload

    def export_sea(self, handle: str, filename: str) -> Dict[str, Any]:
        """Export a simulation result to a ``.sea`` file.

        Parameters
        ----------
        handle : str
            WFData / HAADFData / TACAWData handle.
        filename : str
            Workspace-relative ``.sea`` output path.

        Returns
        -------
        dict
            Handle and written path.

        Raises
        ------
        KeyError
            If the handle is unknown.
        TypeError
            If the object has no ``to_sea`` support.
        ImportError
            If sea-eco is not installed.
        """
        obj = self._get(handle)
        if not hasattr(obj, "to_sea"):
            raise TypeError(
                f"Handle {handle!r} is {type(obj).__name__}, which has no .sea export; "
                "export WFData, HAADFData, or TACAWData handles."
            )
        target = self._workspace_path(filename if filename.endswith(".sea") else f"{filename}.sea")
        obj.to_sea(str(target))
        return {"handle": handle, "sea_path": str(target)}

    # ------------------------------------------------------------------
    # Formatting / registry internals
    # ------------------------------------------------------------------

    def format_response(self, payload: Any, response_format: ResponseFormat) -> str:
        """Format a tool payload as JSON or simple Markdown.

        Parameters
        ----------
        payload : Any
            JSON-safe-able payload from a service method.
        response_format : ResponseFormat
            Requested output format.

        Returns
        -------
        str
            Rendered response text.
        """
        payload = self._json_safe(payload)
        if response_format == ResponseFormat.JSON:
            return json.dumps(payload, indent=2, sort_keys=True)
        lines = ["# Result", ""]
        lines.extend(self._markdown_lines(payload, indent=0))
        return "\n".join(lines).rstrip() + "\n"

    def _register(self, obj: Any, preferred: Optional[str] = None) -> Dict[str, Any]:
        """Register an object and return its handle plus summary.

        Parameters
        ----------
        obj : Any
            Object to register.
        preferred : str | None, optional
            Preferred label for the handle.

        Returns
        -------
        dict
            ``{"handle": ..., **summary}``.
        """
        handle = self._make_handle(obj, preferred=preferred)
        self._objects[handle] = obj
        return {"handle": handle, **self._summary(obj)}

    def _make_handle(self, obj: Any, preferred: Optional[str] = None) -> str:
        """Build a unique ``Type:label`` handle for an object.

        Parameters
        ----------
        obj : Any
            Object being registered.
        preferred : str | None, optional
            Preferred label.

        Returns
        -------
        str
            Unique handle string.
        """
        prefix = type(obj).__name__
        label = "".join(c if c.isalnum() or c in "-_." else "-" for c in (preferred or prefix))[:60] or prefix
        candidate = f"{prefix}:{label}"
        if candidate not in self._objects:
            return candidate
        return f"{candidate}-{uuid4().hex[:8]}"

    def _get(self, handle: str) -> Any:
        """Return a registered object by handle.

        Parameters
        ----------
        handle : str
            Registered handle.

        Returns
        -------
        Any
            The registered object.

        Raises
        ------
        KeyError
            If the handle is unknown; the message lists known handles.
        """
        if handle not in self._objects:
            known = ", ".join(sorted(self._objects)) or "(none)"
            raise KeyError(f"Unknown handle: {handle!r}. Known handles: {known}")
        return self._objects[handle]

    def _require_type(self, handle: str, expected: type) -> Any:
        """Return a handle's object and validate its type.

        Parameters
        ----------
        handle : str
            Registered handle.
        expected : type
            Required object type.

        Returns
        -------
        Any
            The registered object.

        Raises
        ------
        KeyError
            If the handle is unknown.
        TypeError
            If the object has a different type.
        """
        obj = self._get(handle)
        if not isinstance(obj, expected):
            raise TypeError(f"Handle {handle!r} is {type(obj).__name__}, expected {expected.__name__}.")
        return obj

    def _workspace_path(self, filename: str) -> Path:
        """Resolve a workspace-relative artifact path safely.

        Parameters
        ----------
        filename : str
            Workspace-relative path.

        Returns
        -------
        pathlib.Path
            Resolved path inside the workspace (parents created).

        Raises
        ------
        ValueError
            If the path escapes the workspace.
        """
        target = (self.workspace / filename).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"Path {filename!r} escapes the workspace {self.workspace}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _atom_symbols(trajectory: Any) -> List[str]:
        """Return element symbols for a trajectory's atom types.

        Normalizes the atom_types str/int gotcha: string types pass through,
        integer types are treated as atomic numbers.

        Parameters
        ----------
        trajectory : Trajectory
            Trajectory whose ``atom_types`` to normalize.

        Returns
        -------
        list[str]
            Element symbols, one per atom.

        Raises
        ------
        ValueError
            If an integer type is not a valid atomic number.
        """
        from ..multislice.potentials import _ELEMENTS

        symbols: List[str] = []
        for atom_type in trajectory.atom_types:
            if isinstance(atom_type, (str, np.str_)):
                symbols.append(str(atom_type))
            else:
                z = int(atom_type)
                if not 1 <= z <= len(_ELEMENTS):
                    raise ValueError(
                        f"atom type {z} is not a valid atomic number; pass atom_mapping when loading LAMMPS files."
                    )
                symbols.append(_ELEMENTS[z - 1])
        return symbols

    def _trajectory_to_ase(self, trajectory: Any):
        """Convert a trajectory's first frame to ASE atoms with symbol types.

        Parameters
        ----------
        trajectory : Trajectory
            Source trajectory.

        Returns
        -------
        ase.Atoms
            First frame with periodic boundary conditions.

        Raises
        ------
        ValueError
            If atom types cannot be resolved to element symbols.
        """
        from ase import Atoms

        symbols = self._atom_symbols(trajectory)
        return Atoms(symbols, positions=trajectory.positions[0], cell=trajectory.box_matrix, pbc=True)

    @staticmethod
    def _save_heatmap(array: np.ndarray, filename: str, title: str, xlabel: str, ylabel: str) -> None:
        """Save a 2-D array as a PNG heatmap without opening a window.

        Parameters
        ----------
        array : numpy.ndarray
            2-D intensity array.
        filename : str
            Output PNG path.
        title : str
            Plot title.
        xlabel, ylabel : str
            Axis labels.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the array is not 2-D.
        """
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        if np.asarray(array).ndim != 2:
            raise ValueError("heatmap rendering needs a 2-D array")
        fig, ax = plt.subplots()
        ax.imshow(np.abs(np.asarray(array)), cmap="inferno", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _summary(self, obj: Any) -> Dict[str, Any]:
        """Build a type-specific summary of a registered object.

        Parameters
        ----------
        obj : Any
            Registered object.

        Returns
        -------
        dict
            Summary with at least a ``type`` key.
        """
        type_name = type(obj).__name__
        if type_name == "Trajectory":
            unique_types = sorted({str(t) for t in obj.atom_types})
            return {
                "type": type_name,
                "n_frames": int(obj.n_frames),
                "n_atoms": int(obj.n_atoms),
                "atom_types": unique_types[:12],
                "box_diag_A": [round(float(v), 4) for v in np.diag(obj.box_matrix)],
                "extent_A": [round(float(v), 4) for v in obj.extent],
                "timestep_ps": float(obj.timestep),
            }
        if type_name == "MultisliceCalculator":
            return {
                "type": type_name,
                "grid": {"nx": obj.nx, "ny": obj.ny, "n_slices": obj.nz},
                "n_probes": getattr(obj, "n_probes", len(obj.probe_positions)),
                "aperture_mrad": obj.aperture,
                "voltage_eV": obj.voltage_eV,
                "sampling_A": obj.sampling,
                "device": str(obj.device),
            }
        if type_name == "WFData":
            return {
                "type": type_name,
                "shape_probe_time_kx_ky_layer": list(np.shape(obj._array)),
                "n_probes": len(obj.probe_positions),
                "n_frames": int(len(obj.time)) if obj.time is not None else 1,
                "kx_range_invA": [float(obj.kxs.min()), float(obj.kxs.max())],
            }
        if type_name == "HAADFData":
            shape = list(np.shape(obj.data)) if obj.data is not None else None
            return {"type": type_name, "image_shape": shape, "computed": obj.data is not None}
        if type_name in ("TACAWData", "SEDData"):
            frequencies = obj.frequencies
            return {
                "type": type_name,
                "shape_probe_freq_kx_ky": list(np.shape(obj.data)),
                "frequency_range_THz": [float(frequencies.min()), float(frequencies.max())],
            }
        if isinstance(obj, np.ndarray):
            return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        return {"type": type_name}

    def _json_safe(self, value: Any) -> Any:
        """Convert a payload into JSON-serializable primitives.

        Large arrays are summarized instead of inlined to keep responses
        compact.

        Parameters
        ----------
        value : Any
            Payload value.

        Returns
        -------
        Any
            JSON-safe equivalent.
        """
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            if value.size <= 64:
                return self._json_safe(value.tolist())
            return {"shape": list(value.shape), "dtype": str(value.dtype), "note": "array summarized; register/export for full data"}
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.complexfloating, complex)):
            return {"re": float(np.real(value)), "im": float(np.imag(value))}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        return value

    def _markdown_lines(self, payload: Any, indent: int) -> List[str]:
        """Render a JSON-safe payload as indented Markdown bullet lines.

        Parameters
        ----------
        payload : Any
            JSON-safe payload.
        indent : int
            Current indentation level.

        Returns
        -------
        list[str]
            Markdown lines.
        """
        pad = "  " * indent
        lines: List[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}- **{key}**:")
                    lines.extend(self._markdown_lines(value, indent + 1))
                else:
                    lines.append(f"{pad}- **{key}**: {value}")
        elif isinstance(payload, list):
            for value in payload:
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.extend(self._markdown_lines(value, indent + 1))
                else:
                    lines.append(f"{pad}- {value}")
        else:
            lines.append(f"{pad}{payload}")
        return lines
