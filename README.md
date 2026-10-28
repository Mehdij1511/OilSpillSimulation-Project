# Oil Spill Simulation Project

A computational project developed during INF202 at NMBU to model how oil spreads in a coastal environment and estimate its impact on a defined fishing grounds area.

## About this project

This project was my first hands-on experience with scientific computing and numerical modeling. I built a simplified 2D oil spill simulation that represents a coastal region as a triangular mesh, tracks oil distribution over time, and visualizes how the spill evolves through the domain.

The goal was not to build a perfect ocean model, but to create a clear, practical simulation that demonstrates how computational methods can help analyze environmental risk and support decision-making.

---

## Problem

Oil spills can cause severe ecological and economic damage. Once a spill occurs, the oil moves with the surrounding flow and may spread across large areas, threatening marine life, coastal regions, and fishing zones.

The challenge was to answer a few practical questions:
- Where does the oil go?
- How fast does it spread?
- How much reaches a sensitive area?
- Can this be visualized in a way that is understandable and useful?

This project was designed to explore those questions using a simplified but meaningful simulation.

---

## My approach

I modeled the environment as a triangular mesh and treated each triangle as a small computational cell. Each cell stores how much oil is currently present, and neighboring cells exchange oil based on a simple velocity field and shared geometry.

The simulation:
1. Loads a mesh from a `.msh` file
2. Initializes oil near a source point
3. Computes neighbor relationships
4. Updates the oil distribution over time
5. Measures the amount of oil inside a defined fishing grounds region
6. Saves plots and output videos for analysis

This approach turns the real-world phenomenon into a readable, testable computational model.

---

## What the project does

- Reads triangular mesh geometry from a mesh file
- Simulates the movement of oil across the domain
- Uses a velocity field to drive transport between adjacent cells
- Tracks oil amount over time
- Identifies whether oil enters a specified protected area
- Saves image frames and an animation video
- Produces a log of simulation results for analysis

---

## Why this matters

This project is a good example of how programming and scientific thinking can be combined to solve a real-world problem. Instead of just observing an environmental process, we can model it, study its behavior, and quantify its potential impact.

For me, this project was valuable because it showed how simulations can be used as decision-support tools in fields like environmental science, risk analysis, and engineering.

---

## Technologies used

- Python
- NumPy
- meshio
- Matplotlib
- OpenCV
- TOML configuration
- pathlib and argparse
- logging for output tracking

---

## Project workflow

```text
Load mesh
   ↓
Initialize oil source
   ↓
Compute neighbors
   ↓
Simulate time-stepping
   ↓
Compute oil in fishing grounds
   ↓
Save visual outputs and logs
```

---

## Key files

- `main.py` — main simulation entry point
- `src/Simulation/Simulator.py` — simulation logic and time stepping
- `src/Simulation/mesh.py` — mesh loading and cell creation
- `src/Simulation/Visualizer.py` — plotting and video generation
- `input.toml` — configuration for a simulation run
- `bay.msh` — sample mesh used by the project
- `Report.pdf` — project documentation/report

---

## Visual results

The project produces image frames and animation output to visualize how the spill evolves over time.

### Initial state
![Initial oil spill simulation](https://raw.githubusercontent.com/Mehdij1511/OilSpillSimulation-Project/6a26bed50b39df0b333c722c5659e397230ba951/output_t0%20to%20t0.5%20-%20500n/img/plot_0.000.png)

### Intermediate spread
![Oil spread across the mesh](https://raw.githubusercontent.com/Mehdij1511/OilSpillSimulation-Project/6a26bed50b39df0b333c722c5659e397230ba951/output_t0%20to%20t0.5%20-%20500n/img/plot_0.250.png)

### Later stage
![Oil reaching the fishing grounds region](https://raw.githubusercontent.com/Mehdij1511/OilSpillSimulation-Project/6a26bed50b39df0b333c722c5659e397230ba951/output_t0%20to%20t0.5%20-%20500n/img/plot_0.500.png)

### Simulation log
The project also records the evolution in a log file, which captures the changing state of the oil distribution during the simulation.

[Simulation log](https://github.com/Mehdij1511/OilSpillSimulation-Project/blob/6a26bed50b39df0b333c722c5659e397230ba951/output_t0%20to%20t0.5%20-%20500n/t0%20to%20t0.5%20-%20500n.log)

---

## Example output animation

A recorded simulation output is included in the repository and can be used for demonstrations or report visuals.

```text
output_t0 to t0.5 - 500n/output.avi
```

This gives a real example of the simulation running over time and provides a useful visual artifact for presentations.

---

## What I learned

This project taught me a lot about:
- numerical simulation
- mesh-based modeling
- time-stepping methods
- data visualization
- environmental risk analysis
- connecting mathematical ideas to practical code

It was a strong introduction to scientific computing, and it helped me better understand how computational models are used to study real-world systems.

---

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --config input.toml
```

This runs the simulation using the example configuration file and generates outputs in the project directory.

---

## Why this project is important to me

This project represents one of my first steps toward building computational tools that solve meaningful problems. It combines programming, mathematics, environmental thinking, and visualization into a single, practical workflow.

It was not just a coding exercise — it was a way of turning a real-world issue into a model, an analysis, and a visual story.

---

## Summary

This oil spill simulation project models how an oil slick spreads across a spatial domain and estimates its impact on a sensitive fishing area. It uses a mesh-based computational approach and produces visual outputs that make the results understandable and useful.

It is a simple but effective example of how code can be used to simulate and study complex environmental processes.

---

## Project structure

```text
OilSpillSimulation-Project/
├── main.py
├── input.toml
├── requirements.txt
├── bay.msh
├── README.md
├── Report.pdf
├── src/
│   └── Simulation/
│       ├── Simulator.py
│       ├── Visualizer.py
│       ├── mesh.py
│       └── cells.py
├── docs/
│   └── images/
├── output_.../
└── tests/
```

---

## Final note

This project is a good example of applied programming and modeling in a real-world context. It blends technical implementation with environmental problem-solving and helped me build confidence in using code as a tool for analysis and communication.
