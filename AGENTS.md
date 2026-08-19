# PySlice Agent Notes

PySlice (import `pyslice`) is the pySEA ecosystem's GPU-accelerated
multislice simulation engine: TEM/STEM/4D-STEM imaging and diffraction, plus
vibrational EELS via the TACAW method computed from MD trajectories with
universal ML potentials (ORB/FAIRChem). Simulation outputs subclass sea-eco
`Signal` and export calibrated `.sea` files via `.to_sea()`.

## Collaboration protocol

Two contributors (Ondrej and Eric), each working with Claude. PySlice does
not yet carry its own `ai_wiki/` slice, so contributor notes for PySlice work
live at the ecosystem level:
`sea-ecosystem/src/pySEA/ai_wiki/ecosystem/notes/<ondrej|eric>/`. Follow the
protocol in `sea-ecosystem/CLAUDE.md`: pull → read the other contributor's
LOG → write a TODO note with a meaningful kebab-case branch name →
`[Under Construction]` LOG entry → small commits, push and pull between them.

## Install

```bash
pip install -e .                 # core: ase, numpy, scipy, matplotlib, tqdm, h5py, ovito
pip install -e ".[fast]"         # + torch GPU acceleration
pip install -e ".[md]"           # + orb-models ML potentials (pins Python 3.12)
pip install -e ".[mcp]"          # + MCP server deps (mcp, pydantic)
pip install -e ".[sea]"          # + sea-eco for .to_sea() export
```

Note: **OVITO is not on normal PyPI** (`--find-links https://www.ovito.org/pip/`);
it is only needed for the XYZ/LAMMPS/ASE-traj loading path — CIF loading and
everything downstream work without it. `requires-python >= 3.12` (driven by
the MD stack).

## Pipeline (mental model)

```
Loader → Trajectory → (optional MD: ORBMDCalculator/FAIRChemMDCalculator)
       → MultisliceCalculator.setup(...).run() → WFData
       → {HAADFData, TACAWData} → .to_sea()
```

## Repo map

- `src/pyslice/io/loader.py` — `Loader`: CIF (via ASE), XYZ/LAMMPS/traj (via
  OVITO), in-memory ASE `Atoms`; caches `.npy` siblings next to sources.
- `src/pyslice/io/databases.py` — Materials Project + COD search →
  CIF retrieval (stdlib urllib only; MP key via `PYSLICE_MP_API_KEY`).
- `src/pyslice/multislice/trajectory.py` — `Trajectory` (positions,
  velocities, atom_types, box_matrix, timestep in **ps**) + transforms
  (tile, rotate_to, tilt, slice, frozen-phonon displacements).
- `src/pyslice/multislice/calculators.py` — `MultisliceCalculator`
  (setup/run), `SEDCalculator`.
- `src/pyslice/multislice/multislice.py` — `Probe`, `PrismProbe`,
  `Propagate`, aberrations (abTEM Cnm convention).
- `src/pyslice/multislice/potentials.py` — Kirkland potentials,
  `grid_from_trajectory`.
- `src/pyslice/md/molecular_dynamics.py` — ASE MD with ML potentials
  (timestep in **fs**; equilibration → production; no cancel hook).
- `src/pyslice/postprocessing/{wf_data,haadf_data,tacaw_data}.py` — result
  Signals with `.to_sea()`/`.load()`.
- `src/pyslice/data/pyslice_serial.py` — the sea-eco serialization bridge
  (`PySliceSerial` mixin importing `pySEA.sea_eco.architecture.base_structure`).
- `src/pyslice/backend.py` — numpy/torch backend seam (`make_backend`,
  `PYSLICE_DEVICE`, `PYSLICE_BACKEND=numpy`).
- `src/pyslice/mcp/{service,server}.py` — MCP surface (`python -m pyslice.mcp`).
- `skills/` — agent skills; `.claude/agents/` — subagents; `tests/` —
  numbered pytest files; `examples/` — runnable technique scripts.

## AI tools

- **MCP:** `python -m pyslice.mcp [--workspace DIR]` — `pyslice_*` tools;
  call `pyslice_get_conventions` first. Thin server over
  `pyslice.mcp.service.PySliceService` (mirrors `pySEA.sea_eco.mcp`).
- **Skills:** `skills/pyslice` (umbrella), `multislice-imaging`,
  `tacaw-phonons`, `md-setup`, `simulation-parameter-selection` (the
  parameter physics — single source of truth), `structure-retrieval`
  (Materials Project / COD → CIF).
- **Subagents:** `.claude/agents/` — `multislice-parameter-advisor`,
  `simulation-runner`, `structure-builder`, `experiment-matcher`,
  `tacaw-analyst`, plus dev-time `pyslice-integration-reviewer` and
  `sea-data-curator`.

Keep the layering: MCP tools are the hands, skills are the know-how,
subagents are the workers. The parameter physics lives ONCE, in
`skills/simulation-parameter-selection/SKILL.md`, implemented by
`pyslice_suggest_parameters`; reference it, never duplicate it.

## Core rules

- **Units:** Å for lengths, mrad for angles (aperture/detectors), eV for
  voltage, **ps** for trajectory frame spacing, **fs** for the MD
  integration timestep, THz for TACAW frequencies, 1/Å for k-axes.
- **`aperture=0` = parallel beam (TEM); `>0` = convergent STEM probe.**
- **`atom_types` gotcha:** element-symbol strings on the ASE/CIF path,
  integer type ids on the OVITO path; `to_ase()` assumes strings. Normalize
  with `atom_mapping` at load time; `Potential._resolve_z` handles both.
- **Structure edits are Trajectory transforms** (tile/rotate/tilt/crop/
  frozen-phonon), never `setup()` arguments; transforms return new objects.
- **Every frame is propagated** — frame count = frozen-phonon configs or MD
  snapshots.
- **Blocking compute:** `MultisliceCalculator.run()` and MD `run()` block
  with tqdm and have no cancel hook; native/GPU code can hard-crash the
  process. Run them behind a subprocess/job boundary in host applications.
- **Keep the `.sea` bridge stable:** `pyslice_serial.py` and the
  `_sea_config` dicts define the on-disk format; verify against *current*
  sea-eco (`Dimensions(..., det_dimensions=...)`) and round-trip test any
  change.
- **Caches:** wavefunctions under `psi_data/<backend>_<hash>/`, loader
  arrays as `.npy` siblings of the source file. Redirect to scratch dirs in
  managed environments.
- **Backend seam:** get array ops from `pyslice.backend.make_backend()`;
  never import torch directly in simulation code.

## Testing

```bash
python -m pytest tests/ -q
```

Tests are numbered `NN_topic.py`. GPU-, OVITO-, and network-dependent tests
gate themselves; `tests/27_mcp_service.py` and `tests/28_databases.py` run
CPU-only with mocked HTTP (set `PYSLICE_DB_LIVE_TESTS=1` for the live COD
round-trip). `tests/16_sea_eco_integration.py` exercises the `.sea` bridge
and needs sea-eco installed.

## Style rules

- NumPy docstring style for all new Python callables (public, private,
  dunder, properties). Documentation quality is part of code correctness.
- Prefer explicit type annotations; `Literal` over `str` for fixed value
  sets; `Sequence` over concrete containers in signatures.
- No new dataclasses (existing `Trajectory` predates the rule; do not copy
  the pattern into new code or other repos).
- Errors must be actionable: name the fix, the env var, or the install
  command.
- Prefer additive changes; `.sea` files and the packed cache formats are
  compatibility surfaces.

## Ecosystem context

For the full picture of sibling repos and connections, read
`sea-ecosystem/src/pySEA/ai_wiki/ecosystem/CLAUDE.md`. PySlice outputs feed
sea-eco Signals (`.to_sea()`), the planned PoseiTEM Simulation capability,
and PoseidonWeb workspaces. This repo is not yet registered in the ecosystem
wiki (`ai_wiki` slice pending — it requires adopting the shared layout).

## Agent file policy

`CLAUDE.md` is canonical; keep `AGENTS.md` a byte-identical copy
(`cp CLAUDE.md AGENTS.md`). Never edit `AGENTS.md` directly.
