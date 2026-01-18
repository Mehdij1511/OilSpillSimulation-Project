# Oil Spill Simulation Framework

## Overview

A Python framework for simulating 2D oil spill dynamics using a mesh-based approach. The model initializes a Gaussian spill and simulates dispersion based on flux computations, velocity fields, and geometric constraints. It includes a visualizer to generate step-by-step images and video output.

## Quick Start

### 1. Installation

Install the required libraries:

```bash
pip install -r requirements.txt

```



### 2. Run Simulation

Execute the simulation using a configuration file:

```bash
python main.py --config input.toml

```

**Arguments:**

* `-c` / `--config`: Path to the config file (default: "input.toml").


* `--find_all`: Find and run all config files in the main folder.


* `-f` / `--folder`: Specify a folder to search for config files (requires `--find_all`).



## Configuration

Control the simulation via TOML files. Key parameters include:

* **`meshName`**: The mesh file defining the simulation domain.


* **`nSteps`**: Total number of simulation steps (higher = more accuracy).


* **`writeFrequency`**: How often to save frames/states. Set to `0` to disable video.


* **`borders`**: Coordinates defining critical areas (e.g., fishing grounds).


* **`restartFile`**: (Optional) Path to a file to initialize from a saved state.



## Output

For each run, an `output_{logName}` directory is created containing:

* **Simulation state files**: Data for restart or analysis.


* **Images & Video**: Visualizations of the spill evolution.


* **Log file**: Execution details.



## Code Structure

The `Simulation` package is modularized as follows:

* **`Simulator.py`**: Handles core simulation logic, flux calculation, and time stepping.


* **`Visualizer.py`**: Handles plotting and video generation.


* **`mesh.py`**: Processes the mesh and manages cell sorting.


* **`cells.py`**: Defines `Triangle` and `Line` classes and geometric properties.



---
