# Measured-grid stratification pilot

Reproducible pilot for the brief in `../codex_measured_grid_stratification.txt`:
does Dleto's chiseling discover axis transformations and a stratification in
physically measured 3D grids that were not supplied to it?

## Run

From the repository root (Julia 1.12.6, project environment of this repo):

```
julia -t 4 --project=. labs/measured-grids/run_3dfm.jl      # ~1 min
julia -t 4 --project=. labs/measured-grids/run_meshrir.jl   # ~10 min after the download
julia -t 4 --project=. labs/measured-grids/run_haneda.jl    # ~15 min; includes the full 48x51x26 grid
julia -t 4 --project=. labs/measured-grids/probe_meshrir_window.jl   # probe of the one gapped spectrum
```

Each run writes a log, CSV spectra, coefficient curves and the learned axis
transforms to `results/`, and figures to `figures/`.  `common.jl` holds the
method and diagnostics shared by the runs.  `REPORT.md` is the write-up.

## Data (not in git; `labs/data/` is ignored)

| dataset | file | provenance | check |
|---|---|---|---|
| 3DFM magnetic scan | `labs/data/3dfm/10cmcube_11steps_per_axis_measurement.csv` | https://raw.githubusercontent.com/mri4all/field_mapper_3DFM/main/measurements/10cmcube_11steps_per_axis_measurement.csv (repo MIT) | SHA-256 `a39160b59f6516e1931ce5125886fa8e25f8c718350b2719a1c810d924e53f3f`, 92,563 bytes, 1,333 data rows |
| MeshRIR S1-M3969 v1.2 | `labs/data/meshrir/S1-M3969_npy.zip`, unzipped beside it | https://zenodo.org/records/10852693 (doi:10.5281/zenodo.10852693), CC BY 4.0, Koyama et al. WASPAA 2021 | MD5 `2cb598eb44bb9905560c545db7af3432`, verified |
| Haneda–Ren Part 0, loudspeaker 1 | `labs/data/meshgrid-ir-raw/Raw_A_ir_48k_sp1.h5` | https://zenodo.org/records/19656813 (doi:10.5281/zenodo.19656813), CC BY 4.0, Haneda & Ren | MD5 `4a8c756cadac33326b5bcae02135df95` (Zenodo API), verified; 1,222,807,352 bytes |

Fetch commands:

```
curl -L -o labs/data/3dfm/10cmcube_11steps_per_axis_measurement.csv "https://raw.githubusercontent.com/mri4all/field_mapper_3DFM/main/measurements/10cmcube_11steps_per_axis_measurement.csv"
curl -L -o labs/data/meshrir/S1-M3969_npy.zip "https://zenodo.org/records/10852693/files/S1-M3969_npy.zip?download=1" && (cd labs/data/meshrir && unzip S1-M3969_npy.zip)
curl -L -o labs/data/meshgrid-ir-raw/Raw_A_ir_48k_sp1.h5 "https://zenodo.org/records/19656813/files/Raw_A_ir_48k_sp1.h5?download=1"
```

## Pinned versions

Julia 1.12.6; Dleto at git `3e27c3c` plus the uncommitted additions of HDF5,
FFTW and NPZ to `Project.toml` (this working tree); ITensors v0.9.30,
FFTW v1.10.0, NPZ v0.4.3, CSV v0.10.16, DataFrames v1.8.2, Plots v1.41.6
(GR backend for the figures).  `Manifest.toml` is not committed in this
repository; `Pkg.status()` output is recorded in `REPORT.md`.
