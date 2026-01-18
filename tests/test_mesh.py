import pytest
import numpy as np
from src.Simulation.mesh import _CellFactory, Line, Triangle

@pytest.fixture
def cell_factory():
    cf = _CellFactory()
    cf.register('line', Line)
    cf.register('triangle', Triangle)
    return cf

@pytest.fixture
def sample_points():
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])

@pytest.mark.parametrize("cell_type,expected_class", [
    ('line', Line),
    ('triangle', Triangle)
])
def test_cellfactory_registration(cell_factory, cell_type, expected_class):
    """
    Test the registration of a cell type in the cell factory.
    Args:
        cell_factory: The cell factory instance.
        cell_type: The type of cell to register.
        expected_class: The expected class of the registered cell type.
    """
    assert cell_type in cell_factory._cellTypes
    assert cell_factory._cellTypes[cell_type] == expected_class


@pytest.mark.parametrize("cell_type,point_indices,expected_type,expected_points", [
    ('line', [0, 1], Line, [[0.0, 0.0], [1.0, 0.0]]),
    ('triangle', [0, 1, 2], Triangle, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
])
def test_cellfactory_creation(cell_factory, sample_points, cell_type, point_indices, expected_type, expected_points):
    """
    Test the creation of a cell using the cell factory. Cheks if its the correct type and that points are correctly set to coordinate points
    Args:
        cell_factory: Factory function to create cells.
        sample_points (array): Sample points.
        cell_type (type): Type of the cell to create.
        point_indices (list): Indices of points to use for the cell.
        expected_type (type): Expected type of the created cell.
        expected_points (list): Expected points of the created cell.
    """
    idx = 0
    points = sample_points[point_indices]
    cell = cell_factory(cell_type, point_indices, idx, points)
    assert isinstance(cell, expected_type)
    assert np.array_equal(cell.points, np.array(expected_points))
