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

Requirements: Python 3.11 (uses `tomllib`) or change code to use `tomli` for older Python versions.

Install and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --config input.toml
```

If you prefer wider Python compatibility (3.8–3.10), update `main.py` to use the `tomli` package instead of the stdlib `tomllib` (see Troubleshooting below).

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

Example commands to turn frames into a GIF for embedding in reports:

```bash
# using ImageMagick (install: apt install imagemagick)
convert -delay 5 -loop 0 output_demo/img/plot_*.png docs/images/simulation_demo.gif
```

If you don't have ImageMagick, use FFmpeg to make a video instead:

```bash
ffmpeg -framerate 10 -pattern_type glob -i 'output_demo/img/plot_*.png' -c:v libx264 -pix_fmt yuv420p output_demo/simulation_demo.mp4
```

Suggested README visualization workflow:
1. Run `python main.py -c input.toml` with `writeFrequency` set to produce frames.
2. Convert frames to GIF/MP4 as shown above.
3. Add the generated GIF file to `docs/images/` and reference it in the README:

```markdown
![Simulation demo](docs/images/simulation_demo.gif)
```

---

## Troubleshooting & suggested improvements

- tomllib vs tomli: `main.py` currently uses `tomllib` (Python 3.11+). To run on older python versions, replace the tomllib usage with `import tomli` and `tomli.load()` and add `tomli` to `requirements.txt`.

- Requirements: `requirements.txt` lists unpinned packages. Consider pinning major versions or adding a `pyproject.toml`/`constraints.txt` for reproducibility.

- Large files: `bay.msh` and `Report.pdf` are large blobs — consider moving them to GitHub Releases or use Git LFS so repo clones remain lightweight.

- Tests: `tests/` is present but empty — add pytest-based tests for Mesh reading, neighbor computation, and a short simulation run (deterministic with small mesh) to prevent regressions.

- CI: Add a lightweight GitHub Actions workflow to run lint/pytest on push; include a matrix for Python versions if you broaden toml loader compatibility.

---

## Development notes (for maintainers)

- Relative path handling: `main.py` resolves mesh and restart file paths relative to the TOML directory — keep that in mind when invoking from other working directories.
- Visualizer currently expects oil amounts normalized between 0 and 1 for colormap indexing. If you change initialization or physics, update the color scaling in `Visualizer.create_plot()`.

---

## Contributing

If you'd like me to:

- Add an example GIF to `docs/images` and embed it in this README,
- Modify `main.py` to use `tomli` for backward compatibility,
- Add a small GitHub Actions CI workflow (pytest), or
- Move large assets to `releases/` and add a lightweight example mesh in repo,

tell me which items and I will prepare a single PR with the changes.
