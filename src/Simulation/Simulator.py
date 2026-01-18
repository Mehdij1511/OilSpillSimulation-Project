from .mesh import Mesh
import numpy as np

class simulator:
    def __init__(self, config):
        """
        Initializes the Simulator with the given configuration. And runs the oil initialization.
        Args:
            config (Config): config object for current file
        Attributes:
            mesh (Mesh): Mesh object created using the mesh name from the configuration.
            dt (float): Time step
            tStart (float): Start time of the simulation.
            current_time (float): Current time in the simulation, initialized to the start time.
            Triangles (list): List of triangles in the mesh.
        """
        self.config = config
        self.mesh = Mesh(self.config.meshName)
        self.dt = (self.config.tEnd - self.config.tStart) / self.config.nSteps
        self.mesh.compute_neighbors()
        
        self.tStart = self.config.tStart
        self.current_time = self.tStart
        self.Triangles = self.mesh.get_triangles()
        
        if config.restartFile is None:
            self._initilize_oil_distribution()
        else:
            self._load_restart_file(config.restartFile)


    def _load_restart_file(self, restart_file):
        """
        Loads the simulation state from a restart file and updates the oil amount in each cell.
        Args:
            restart_file (str): Path to the restart file containing the simulation state.
        Raises:
            RuntimeError: If there is an error reading the restart file or parsing its contents.
        """
        try:
            state = {}
            with open(restart_file, 'r') as f:
                for line in f:
                    idx, amount = line.strip().split()
                    state[int(idx)] = float(amount)
        
            for cell in self.Triangles:
                cell.oil_amount = state.get(cell.idx, 0.0)
        except Exception as e:
            raise RuntimeError(f"Failed to load restart file: {e}")

        
    def _initilize_oil_distribution(self):
        """
        Initializes the oil distribution. Based on the distance from the center point x_star. 
        Attributes:
            x_star (np.array): The center point of the oil distribution.
            Triangles (list): List of triangles in the mesh.
        """
        x_star = np.array([0.35, 0.45])
        
        for cell in self.Triangles:
            x = cell.midpoint
            dist_squared = np.sum((x - x_star)**2)
            cell.oil_amount = np.exp(-dist_squared / 0.01)

        
    def _get_velocity(self, x, y):
        """
        Calculate the velocity at a given point (x, y) in the velocity field.
        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
        Returns:
            array: The velocity vector at the given point.
        """
        return np.array([y - 0.2 * x, -x])


    def _compute_flux(self, celle_i, neighbor_i, edge_idx):
        """
        Computes the flux of oil between a cell and its neighbor.
        Args:
            celle_i (Cell): The current cell
            neighbor_i (Cell): on of the neighbors of the current cell
            edge_idx (int): The index of the edge between the current cell and its neighbor.
        Returns:
            float: The computed flux of oil between the current cell and its neighbor.
        """
        v_i = self._get_velocity(celle_i.midpoint[0], celle_i.midpoint[1])
        v_ngh = self._get_velocity(neighbor_i.midpoint[0], neighbor_i.midpoint[1])
        v_avg = 0.5 * (v_i + v_ngh)
        
        scaled_normal = celle_i.get_scaled_normals()[edge_idx]
        dot_product = np.dot(v_avg, scaled_normal)
        # g, dot > 0 -> Means angle is less than 90 degrees (they point in the same direction) Flows from i to ngh or ngh to i
        if dot_product > 0:
            return (-self.dt * celle_i.oil_amount * dot_product) / celle_i._area
        else:
            return (-self.dt * neighbor_i.oil_amount * dot_product) / celle_i._area
        

    def find_shared_edge(self, cell, neighbor):
        """
        Finds the index of the shared edge between a given cell and its neighbor.
        Args:
            cell: current cell to check
            neighbor: neighbor of current cell to check
        Returns:
            int: The index of the shared edge between the cell and its neighbor, or None if no shared edge is found.
        """
        shared_points = set(cell.get_pointIDs()) & set(neighbor.get_pointIDs())

        edges = [
            (cell.get_pointIDs()[0], cell.get_pointIDs()[1]),
            (cell.get_pointIDs()[1], cell.get_pointIDs()[2]),
            (cell.get_pointIDs()[2], cell.get_pointIDs()[0])]

        for edge_idx, (p1, p2) in enumerate(edges):
            if p1 in shared_points and p2 in shared_points:
                return edge_idx


    def step(self):
        """
        Step the simulation forward and incrementing current time step dt
        Updates oil in each cell by total flux. 
        Attributes:
            old_amount (dict): A dictionary of every cell with its oil level
            total_flux (float): The total flux of oil for a given cell from all its neighbors.
            neighbour_idx (int): The index of a neighboring cell.
            neighbour (Cell): The neighboring cell.
            edge_idx (int or None): The index of the shared edge between the current cell and its neighbor.
        """
        old_amount = {cell.idx: cell.oil_amount for cell in self.Triangles}
        for cell in self.Triangles:
            total_flux = 0
            for neighbour_idx in cell.get_neighbors():
                neighbour = self.Triangles[neighbour_idx]
                edge_idx = self.find_shared_edge(cell, neighbour)

                if edge_idx is not None:
                    total_flux += self._compute_flux(cell, neighbour, edge_idx)

            cell.oil_amount = old_amount[cell.idx] + total_flux
        self.current_time += self.dt
    

    def _compute_fishing_grounds(self):
        """
        Computes the total amount of oil fish within the defined fishing grounds.
        Checks if the midpoint of each cell lies within the borders,
        and sums up the oil amount in those cells and multiplies it by the area of the cell.
        Returns:
            float: The total amount of oil fish within the specified fishing grounds.
        """
        total_oil_fish = 0
        x_range = self.config.borders[0]
        y_range = self.config.borders[1]

        for cell in self.Triangles:
            x, y = cell.midpoint
            if (x_range[0] <= x <= x_range[1] and 
                y_range[0] <= y <= y_range[1]):
                total_oil_fish += cell.oil_amount * cell.get_area()
        return total_oil_fish
    

    def get_oil_in_fishing_grounds(self):
        return self._compute_fishing_grounds()


    def get_state(self):
        return {
            cell.idx: cell.oil_amount for cell in self.Triangles
        }
           