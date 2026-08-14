# Oil Spill Simulation Framework

## Overview

A Python 2D mesh-based oil-spill simulator (student project from INF202, NMBU) that initializes a Gaussian spill on a triangular mesh and advances it with flux computations between mesh cells. The project is designed for experimenting with mesh-based advection of a scalar field (oil concentration), visualizing the spill evolution, and measuring contamination in user-defined zones ("fishing grounds").

This README has been expanded with clearer instructions, configuration examples, and visualization guidance so other students and researchers can run experiments and include results in reports.

---

## Features

- Read .msh meshes using meshio and build triangle/line cell objects.
- Initialize an oil spill (Gaussian) or load a restart state.
- Time stepping with per-edge flux computations based on a simple velocity field.
- Save per-step state files and PNG visualizations; optional AVI video creation with OpenCV.
- Compute total oil mass inside a configurable rectangular "fishing grounds" area.

### Notable files
- `main.py` — CLI, config parsing, run loop and output management.
- `src/Simulation/Simulator.py` — simulation algorithm and time stepping.
- `src/Simulation/mesh.py` — mesh reader and cell construction (uses meshio).
- `src/Simulation/cells.py` — Triangle and Line geometry primitives.
- `src/Simulation/Visualizer.py` — plotting and video generation.
- `input.toml` — example configuration file used by the CLI.
- `bay.msh` — example mesh used for tests/demos (large binary).

---

## Quick Start

Requirements: Python 3.11 (uses `tomllib`) or change code to use `tomli` for older Python versions. The repository includes a tomli fallback in `main.py` so it can run on Python 3.8–3.11.

Install and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --config input.toml
```

If you prefer wider Python compatibility (3.8–3.10), the project falls back to `tomli` automatically; ensure `tomli` is present in your environment or in `requirements.txt`.

---

## Configuration (example)

A minimal example TOML configuration to run the simulation. Place this in `input.toml` or pass a custom path with `-c`.

```toml
[settings]
nSteps = 100
tStart = 0.0
tEnd = 1.0

[geometry]
meshName = "bay.msh"
borders = [[0.2, 0.5], [0.2, 0.5]]  # x-range and y-range for fishing grounds

[IO]
logName = "demo"
writeFrequency = 10  # set to 0 to disable images/video
restartFile = ""
```

Key fields:
- `nSteps`: number of time steps (higher gives smaller dt)
- `meshName`: path to the mesh file (relative paths resolved relative to the TOML file)
- `borders`: [[x_min, x_max], [y_min, y_max]] rectangle for measuring oil in fishing grounds
- `writeFrequency`: how often (in steps) to write images/states. 0 disables images/video.

---

## Output and visualization

Running a simulation creates `output_<logName>/` with subfolders:
- `states/` — text files with per-cell oil amounts
- `img/` — PNG frames created with `Visualizer.create_plot()`
- `output.avi` — AVI video created by `Visualizer.create_animation()` (if writeFrequency != 0)
- `<logName>.log` — run log with info messages

### Real simulation output (included)

This repository already contains a recorded run produced by the simulation. The real animation file is:

- output_t0 to t0.5 - 500n/output.avi

You can view it on GitHub (click to download/play):
https://github.com/Mehdij1511/OilSpillSimulation-Project/blob/main/output_t0%20to%20t0.5%20-%20500n/output.avi

If you want to embed a still frame in the README, extract a frame from the AVI and commit it to `docs/images/` (example commands below). The AVI file is a faithful, real output produced by the code — use it as the canonical visualization.


### Visualization gallery (how-to and examples)

- Example thumbnail (real outputs) — to include a still image in the README, extract a frame using ffmpeg and commit it. Example extraction command:

```bash
# extract a representative frame (timestamp 00:00:00.10) and save as PNG
ffmpeg -i "output_t0 to t0.5 - 500n/output.avi" -ss 0.1 -vframes 1 docs/images/simulation_frame.png
```

- To produce a high-quality MP4 from frames (or from the AVI), use:

```bash
# from PNG frames
ffmpeg -framerate 10 -pattern_type glob -i 'output_<logName>/img/plot_*.png' -c:v libx264 -pix_fmt yuv420p docs/images/simulation_demo.mp4

# or re-encode the existing AVI to MP4
ffmpeg -i "output_t0 to t0.5 - 500n/output.avi" -c:v libx264 -crf 23 -preset medium docs/images/simulation_demo.mp4
```

- To create an animated GIF for embedding in the README:

```bash
convert -delay 5 -loop 0 output_<logName>/img/plot_*.png docs/images/simulation_demo.gif
# or use ffmpeg+gifsicle for better quality
ffmpeg -i docs/images/simulation_demo.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" -f gif - | gifsicle --optimize=3 > docs/images/simulation_demo.gif
```

Notes on visualization quality and scaling:
- Visualizer currently expects oil amounts normalized between 0 and 1. If you change initialization or physics, scale values or update the colormap normalization in `src/Simulation/Visualizer.py` (lines around colorbar/Normalize).
- The Visualizer uses matplotlib's `fill` per triangle which is simple and portable but not optimized for very large meshes. For publication-quality images consider rasterizing at higher DPI or using Geo-aware plotting libraries if you expand to real geographic coordinates.

---

## Troubleshooting & suggested improvements

- tomllib vs tomli: `main.py` now prefers `tomllib` (Python 3.11+) and falls back to `tomli` on older interpreters. Consider adding `tomli` to `requirements.txt` for clarity.

- Requirements: `requirements.txt` lists unpinned packages. Consider pinning major versions or adding a `pyproject.toml`/`constraints.txt` for reproducibility.

- Large files: `bay.msh`, `Report.pdf`, and the included `output.avi` are sizable — consider moving them to GitHub Releases or use Git LFS so repo clones remain lightweight.

- Tests: `tests/` is present but empty — add pytest-based tests for Mesh reading, neighbor computation, and a short simulation run (deterministic with small mesh) to prevent regressions.

- CI: Add a lightweight GitHub Actions workflow to run lint/pytest on push; include a matrix for Python versions if you broaden toml loader compatibility.

---

## Development notes (for maintainers)

- Relative path handling: `main.py` resolves mesh and restart file paths relative to the TOML directory — keep that in mind when invoking from other working directories.
- Visualizer currently expects oil amounts normalized between 0 and 1 for colormap indexing. If you change initialization or physics, update the color scaling in `Visualizer.create_plot()`.

---

## Contributing

If you'd like me to:

- Extract a representative frame from the included `output.avi`, commit the PNG to `docs/images/`, and embed it in this README,
- Generate a GIF/MP4 from the included run and commit it to `docs/images/`,
- Add a small GitHub Actions CI workflow (pytest),
- Move large assets to `releases/` and add a lightweight example mesh in repo,

I can prepare a single PR that includes the extracted images and README updates.
