---
name: simulation-parameter-selection
description: Use when choosing or justifying PySlice multislice simulation parameters — sampling, slice thickness, supercell tiling, probe grids, aperture, detector angles, frozen-phonon or MD frame counts, and memory budgets. The single source of truth for the parameter physics behind the pyslice_suggest_parameters MCP tool and the multislice-parameter-advisor subagent.
---

# Simulation Parameter Selection

Turn a structure plus a simulation goal into a concrete, justified PySlice
parameter set. Every rule here is what `pyslice_suggest_parameters`
implements; agents editing code or answering "why this value?" should cite
these rules rather than re-deriving them.

## Read First

- `skills/pyslice/SKILL.md` — units and workflow context
- The structure itself: cell extent, atom count, frame count drive everything
  (`Trajectory.extent`, `.n_atoms`, `.n_frames`)

## Core Rules (the physics)

All lengths in Å, angles in mrad (θ below in radians), voltage in eV.

1. **Wavelength.** λ = 12.2639 / √(V + 0.97845×10⁻⁶ V²).
   100 kV → 0.0370 Å; 60 kV → 0.0487 Å; 200 kV → 0.0251 Å.

2. **Real-space sampling ← largest scattering angle.** The multislice
   band limit keeps k ≤ 1/(3·sampling), so to represent scattering out to
   θ_max: **sampling ≤ λ / (3·θ_max)**. Pick θ_max as 1.2× the ADF outer
   angle for HAADF, ≥3× the aperture for 4D-STEM, or the largest diffraction
   angle of interest. Finer sampling is never wrong, only slower (cost ∝
   N² log N per slice).

3. **Slice thickness.** Target ~0.5 Å; must divide the cell height along the
   beam axis evenly: `n = round(h/0.5); slice_thickness = h/n`. Thicker
   slices (1–2 Å) are acceptable for light elements and low voltages; halve
   the thickness and re-run to check convergence of the observable.

4. **k-space resolution ← lateral cell size.** Δk = 1/L. For a convergent
   probe the aperture disk (radius θ_ap/λ) should be sampled by ≥5 points:
   **L ≥ 5λ/θ_ap**. For dispersion work Δk must resolve the Brillouin-zone
   path — more unit cells = finer k-grid.

5. **Probe wraparound ← lateral cell size.** Periodic boundaries wrap the
   probe: keep **L ≥ 4× the probe diameter** d ≈ 1.22 λ/θ_ap. When the cell
   is too small, tile it (`Trajectory.tile_positions`); repeats =
   ceil(L_needed / L_cell) per lateral axis.

6. **Probe step (STEM scanning).** Image Nyquist: **step ≤ λ/(4·θ_ap)** —
   0.37 Å at 100 kV/25 mrad. Grid points per axis = ceil(scan_extent/step).

7. **Aperture.** 0 = parallel beam (TEM imaging, SAED, LACBED background).
   Convergent STEM: 20–35 mrad typical aberration-corrected; use the
   experiment's value when matching data.

8. **Frames.**
   - Static pattern: 1 frame.
   - Thermal diffuse scattering (HAADF/4D-STEM realism): 8–16 frozen-phonon
     configurations (`generate_random_displacements(n, sigma≈0.05–0.1)`).
   - TACAW: MD snapshots. Time Nyquist: **frame spacing Δt ≤ 1/(2·f_max)**
     (30 THz → 16.7 fs). Frequency resolution: **Δf = 1/(N·Δt)** → N ≥
     1/(Δf·Δt). Example: f_max 30 THz, Δf 0.3 THz → Δt=16.7 fs, N=200,
     T_total=3.3 ps. Run production MD in **NVE** for noise-free dynamics.

9. **Memory.** Wavefunction array is complex64:
   bytes ≈ n_probes × n_frames × nx × ny × n_layers × 8. Reduce with
   `max_kx`/`max_ky` crops, `return_layers=None` (HAADF-only, on-the-fly
   ADF), `kth` sparsification, or coarser sampling — in that order of
   preference.

10. **Detector angles (HAADF).** Inner ≥ ~3× aperture to stay dark-field
    (60–90 mrad typical); outer bounded by θ_max from rule 2.

## Natural Language Task Routing

- "what sampling / pixel size do I need" → rule 2 (state λ and θ_max used).
- "how thick should slices be" → rule 3.
- "how big a supercell / how many unit cells" → rules 4–5 (report both
  constraints, take the max).
- "how many probe positions / scan step" → rule 6.
- "how many frozen-phonon configs / MD frames / how long an MD run" → rule 8.
- "will this fit in memory / why is it slow" → rule 9.
- Any full parameter set request → apply every applicable rule, return the
  set plus one-line justifications (this is exactly what the
  `pyslice_suggest_parameters` MCP tool returns).

## Python API Examples For Agents

```python
import math

def wavelength_A(voltage_eV: float) -> float:
    return 12.2639 / math.sqrt(voltage_eV + 0.97845e-6 * voltage_eV**2)

lam = wavelength_A(100e3)
sampling = lam / (3 * 1.2 * 200e-3)        # represent 1.2 x 200 mrad outer angle
probe_step = lam / (4 * 25e-3)             # 25 mrad probe, image Nyquist
min_L = max(5 * lam / 25e-3, 4 * 1.22 * lam / 25e-3)  # aperture sampling + wraparound
```

Convergence check pattern: run, halve `sampling` (or `slice_thickness`),
re-run, compare the observable; accept when the change is below the noise
floor you care about.

## MCP Tool Use

`pyslice_suggest_parameters` takes a trajectory handle plus a goal
(`diffraction`, `tem_imaging`, `haadf`, `4dstem`, `tacaw`) and returns the
suggested `setup` kwargs, tiling/frame advice, memory estimate, and
per-value justifications. Feed its output into `pyslice_transform_trajectory`
(tiling, frozen_phonon) and `pyslice_setup_multislice`.
