from src.Simulation.Simulator import simulator
from src.Simulation.Visualizer import Visualizer
from pathlib import Path
import tomllib
import logging
import argparse

class SimulationConfig:
    def __init__(self, nSteps, tStart, tEnd, meshName, borders, logName, writeFrequency = None, restartFile = None):
        self.nSteps = nSteps
        self.tStart = tStart
        self.tEnd = tEnd
        self.meshName = meshName
        self.borders = borders
        self.logName = logName
        self.writeFrequency = writeFrequency 
        self.restartFile = restartFile


def read_config(filename):
    """
    Reads and validates a configuration file for a simulation.
    Args:
        filename (str): The path to the configuration file.
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If something is missing in the configuration file.
    Returns:
        SimulationConfig: object with the validated configuration settings.
    """
    if not Path(filename).exists():
        raise FileNotFoundError(f"Config file not found: {filename}")
    with open(filename, 'rb') as f:
        config_dict = tomllib.load(f)

    # Validate required sections exist
    required_sections = ['settings', 'geometry', 'IO']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required section: {section}")

    # Extract sections
    settings = config_dict['settings']
    geometry = config_dict['geometry']
    io = config_dict['IO']

    # Validate required settings
    if 'nSteps' not in settings:
        raise ValueError("Missing: nSteps")
    if 'tEnd' not in settings:
        raise ValueError("Missing: tEnd")
    
    if 'meshName' not in geometry:
        raise ValueError("Missing: meshName")
    if 'borders' not in geometry:
        raise ValueError("Missing: borders")

    if 'tStart' in settings and 'restartFile' not in io:
        raise ValueError("Må ha restartFile om tStart er gitt")

    return SimulationConfig(
        nSteps=settings['nSteps'],
        tStart=settings.get('tStart', 0.0),  # Optional with default 0.0
        tEnd=settings['tEnd'],
        meshName=geometry['meshName'],
        borders=geometry['borders'],
        logName=io.get('logName', 'logfile'),  # Optional with default'logfile'
        writeFrequency=io.get('writeFrequency', 0),  # Optional
        restartFile=io.get('restartFile', None)  # Optional
    )


def find_config_files(folder = None):
    """
    Find all .toml files in the specified folder
    Args:   
        folder (str): Folder to search for .toml files
    Returns:    
        List of Path objects for each .toml file found
    """
    search_path = Path(folder) if folder else Path.cwd()
    if not search_path.exists():
        raise FileNotFoundError(f"Folder not found: {search_path}")
    
    return list(search_path.glob("*.toml"))


def create_output_dir(logName):
    """
    Creates an output directory with subdirectories for states and images.

    Args:
        logName (str): The name to be used in the output directory's name.

    Returns:
        Path: The path to output directory.
    """
    outputdir = Path(f"./output_{logName}")
    outputdir.mkdir(exist_ok=True)
    (outputdir / 'states').mkdir(exist_ok=True)
    (outputdir / 'img').mkdir(exist_ok=True)
    return outputdir


def parse_input():
    """
    parse arguments with options:
    - `-c` or `--config_file`: Path to the config file (default "input.toml").
    - `--find_all`: Find and run all config files in the main folder.
    - `-f` or `--folder`: Specify the folder to search for config files (requires `--find_all`).
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    Raises:
        ArgumentError: If `-f` is provided without `--find_all`.
    """
    parser = argparse.ArgumentParser("Run oil spill simulation")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--config_file", default="input.toml", help="Path to config file")
    parser.add_argument("--find_all", action="store_true", help="Find and run all config files in main folder")
    parser.add_argument("-f", "--folder", help="Specify folder to search for config files")

    args = parser.parse_args()
    if args.folder and not args.find_all:
        parser.error("-f requires --find_all")

    return args


def run_simulation(config_file):
    """
    Runs the oil spill simulation based on the provided configuration file.
    Args:
        config_file (str): Path to the configuration file.
    Raises:
        Exception: If any error occurs during the simulation.
    """
    try:
        config = read_config(config_file)
        print(f"Creating output directory for {config.logName}...")
        outputdir = create_output_dir(config.logName)

        sim = simulator(config)
        vis = Visualizer(sim.mesh.get_triangles(), config, outputdir)

        logging.root.handlers.clear() # CLEAR HANDLER FOR NEW LOGGING FILE FOR NEXT CONFIG
        logger = logging.getLogger()
        handler = logging.FileHandler(outputdir / f"{config.logName}.log", mode='w')
        formatter = logging.Formatter('%(asctime)s- %(levelname)s - %(message)s', datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        string = "Configuration settings:"
        for key, value in config.__dict__.items():
            if value is not None:
                string += (f"\n\t{key}: {value}")
        logging.info(string)


        while sim.current_time <= config.tEnd:
            oil_distribution = sim.get_state()

            if config.writeFrequency == 0:
                pass
            elif round(int(sim.current_time / sim.dt)) % config.writeFrequency == 0:

                #  1 SAVE STATES
                state_path = outputdir / 'states' / f"state_{sim.current_time:.3f}.txt"
                with open(state_path, 'w') as f:
                    for cell_idx, oil_amount in oil_distribution.items():
                        f.write(f"{cell_idx} {oil_amount}\n")

                #  2 SAVE PLOT IMAGES
                plot_path = outputdir / 'img' / f"plot_{sim.current_time:.3f}.png"
                vis.create_plot(oil_distribution, sim.current_time, plot_path)
    
                # 3 CREATE LOG
                logging.info(f"At Time: {sim.current_time:.3f}/{config.tEnd:.3f}: Oil in fishing grounds: {sim.get_oil_in_fishing_grounds():.3e}")
            sim.step()

        
        # SAVE FINAL STEP:
        state_path = outputdir / 'states' / f"state_{sim.current_time:.3f}.txt"
        with open(state_path, 'w') as f:
            for cell_idx, oil_amount in oil_distribution.items():
                f.write(f"{cell_idx} {oil_amount}\n")
        plot_path = outputdir / 'img' / f"plot_{sim.current_time:.3f}.png"
        vis.create_plot(oil_distribution, sim.current_time, plot_path)        
        logging.info(f"At Time: {sim.current_time:.3f}/{config.tEnd:.3f}: Oil in fishing grounds: {sim.get_oil_in_fishing_grounds():.3e}")

        if config.writeFrequency != 0:
            images = list(Path(outputdir / 'img').glob("*.png"))
            vis.create_animation(images, config.writeFrequency)

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
    

if __name__ == "__main__":
    args = parse_input()
    try:
        if args.find_all:
            config_files = find_config_files(args.folder)
            if not config_files:
                print(f"No config files found in {args.folder or 'current directory'}")

            print(f"Found {len(config_files)} config file(s)")
            for config_file in config_files:
                print(f"\nProcessing {config_file.name}")
                run_simulation(config_file)
            print(f"\nCompleted {len(config_files)} simulations")
            
        else: 
            args.config_file
            config_path = Path(args.config_file)
            print(f"\nProcessing {args.config_file}")
            run_simulation(config_path)
            print(f"\nCompleted simulation for {args.config_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)