from .cells import Triangle, Line
import meshio


class _CellFactory:
    """
    A factory class for creating cell objects.
    Attributes:
        _cellTypes (dict): A dictionary to store registered cell types.
    Methods:
        __init__(): Initializes the Cell factory with an empty cell types dictionary.
        register(key, name): Registers a cell type with a given key and name.
        __call__(key, point_indices, idx, points): Creates and returns a cell object of the registered type.
    """
    def __init__(self):
        self._cellTypes = {}
    def register(self, key, name):
        self._cellTypes[key] = name
    def __call__(self, key, point_indices, idx, points):
        return self._cellTypes[key](point_indices, idx, points)


class Mesh:
    def __init__(self,mshname):
        """
        Initializes the mesh object by reading mesh data from a file and creating
        line and triangle cells from 'Cell'.
        Args:
            mshname (str): The name of the mesh file to read.
        Attributes:
            _points (array): points from the mesh file.
            _lines (list): List of line cells created from the mesh file.
            _triangles (list): List of triangle cells created from the mesh file.
        """
        msh = meshio.read(mshname)
        self._points = msh.points
        
        cf = _CellFactory()
        cf.register('line', Line)
        cf.register('triangle', Triangle)

        self._lines = []
        self._triangles = []
        for CellForType in msh.cells:
            cellType = CellForType.type
            if cellType not in cf._cellTypes:
                continue
            cellPoints = CellForType.data
            for point_indices in cellPoints:
                idx = len(self._triangles) if cellType == 'triangle' else len(self._lines)
                points = self._points[point_indices][:, :2]  # removes z-coordinates

                cell = cf(cellType, point_indices, idx, points)
                
                if cellType == 'triangle':
                    self._triangles.append(cell)
                else:
                    self._lines.append(cell)


    def get_triangles(self):
        return self._triangles


    def compute_neighbors(self):
        """
        Computes the neighbors for each cell in the mesh.
        Takes every cell in mesh and calls 'compute_neighbors'
        """
        for cell in self._triangles:
            cell.compute_neighbors(self._triangles)