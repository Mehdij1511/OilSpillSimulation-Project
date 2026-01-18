import numpy as np


class Cell():
    def __init__(self, point_indices, idx, points):
        """
        Initialize a Cell object.
        Attributes:
            point_indices (list): List of point IDs that make up the cell.
            idx (int): cell id
            points (array): Coordinates of the points that make up the cell.
        """
        self._pointIDs = point_indices
        self.idx = idx
        self._neighbors = []
        self.points = points

        self.midpoint = np.mean(points, axis=0)
        self._edges = None
        self._normals = None
        self._scaled_normals = None
        self._area = None
        self._compute_geometry()


    def compute_neighbors(self, cells):
        """
        Calculate neighboring cells based on shared points.
        Goes through all cells and checks if cells are neighbors by
        checking if its points are shared (&) with the cell given to check with.
        Then stores all neighbors in a list.
        Args:
            cells (list): A list of cell objects to check for neighbors.
        """
        self._neighbors = []
        for cell in cells:

            if not isinstance(cell, Triangle): continue 

            if cell.idx != self.idx: 
                common_points = set(self._pointIDs) & set(cell._pointIDs)
                if len(common_points) == 2:  # Cells share an edge
                    self._neighbors.append(cell.idx)


    def get_neighbors(self):
        return self._neighbors
    
    
    def get_pointIDs(self):
        return self._pointIDs


class Line(Cell):
    def __init__(self, point_indices, idx, points):
        super().__init__(point_indices, idx, points)
        self.oil_amount = 0.0

    # Here because inherited from Cell
    def _compute_geometry(self):
        pass
    def __str__(self):
        return f"Line (ID: {self.idx}) Neighbors: {self._neighbors}"


class Triangle(Cell):
    def __init__(self, point_indices, idx, points):
        super().__init__(point_indices, idx, points)
        self.oil_amount = 0.0


    def _compute_geometry(self):
        """
        Calculate all variables for the triangle being made.
        Attributes:
            self._edges (list): List of vectors representing the edges of the triangle.
            self._normals (list): List of unit vectors perpendicular to each edge.
            self._scaled_normals (list): List of normals scaled by the length of the corresponding edge.
            self._area (float): The area of the triangle.
        """
        # EDGES
        self._edges = [
            self.points[1] - self.points[0],
            self.points[2] - self.points[1],
            self.points[0] - self.points[2]]
        
        # NORMALS
        self._normals = []
        for i in range(3):  # For every edge on the triangle
            if i == 0:
                edge = self._edges[0]
                edge_midpoint = (self.points[0] + self.points[1]) / 2
            elif i == 1:
                edge = self._edges[1]
                edge_midpoint = (self.points[1] + self.points[2]) / 2
            else:
                edge = self._edges[2]
                edge_midpoint = (self.points[2] + self.points[0]) / 2

            normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)

            center_to_midpoint = edge_midpoint - self.midpoint
            if np.dot(normal, center_to_midpoint) < 0:
                normal = -normal
            self._normals.append(normal)

        # SCALED NORMALS
        self._scaled_normals = [n * np.linalg.norm(e) for n, e in zip(self._normals, self._edges)]

        # AREA
        edge1 = self._edges[0]
        edge2 = self._edges[1]
        self._area = 0.5 * abs(edge1[0]*edge2[1] - edge1[1]*edge2[0])

    def get_area(self):
        return self._area
    

    def get_scaled_normals(self):
        return self._scaled_normals


    def __str__(self):
        return f"Triangle (ID: {self.idx}) Neighbors: {self._neighbors}"