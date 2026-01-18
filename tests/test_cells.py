import pytest
import numpy as np

@pytest.mark.parametrize("points,expected_area", [
    (
        np.array([[0, 0], [1, 0], [0, 1]]), 0.5
    ),
    (
        np.array([[0, 0], [2, 0], [0, 2]]), 2.0
    ),
    (
        np.array([[0, 0], [0, 0], [0, 0]]), 0.0
    ),
    (
        np.array([[0, 0], [2, 0], [1, 2]]), 2.0
    ),
    (
        np.array([[1, 1], [3, 1], [2, 3]]), 2.0 
    )])
def test_area_computation(points, expected_area):
    """Test area computation using cross product method for various triangles."""
    edge1 = points[1] - points[0]
    edge2 = points[2] - points[0]
    area = 0.5 * abs(edge1[0]*edge2[1] - edge1[1]*edge2[0])
    assert np.allclose(area, expected_area)


@pytest.mark.parametrize("points,edges", [
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        [np.array([1, 0]), np.array([-1, 1]), np.array([0, -1])]
    ),
    (
        np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]]),
        [np.array([1, 0]), np.array([-0.5, np.sqrt(3)/2]), np.array([-0.5, -np.sqrt(3)/2])]
    )])
def test_normal_computation(points, edges):
    """Test computation of normal vectors for triangle edges. Test for characteristics of normal vectors."""
    midpoint = np.mean(points, axis=0)
    normals = []
    
    for i, edge in enumerate(edges):
        normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
        
        edge_points = [points[i], points[(i+1)%3]]
        edge_midpoint = np.mean(edge_points, axis=0)
        
        # Ensure normal points outward
        center_to_midpoint = edge_midpoint - midpoint
        if np.dot(normal, center_to_midpoint) < 0:
            normal = -normal
        
        normals.append(normal)
        
        # Verify properties
        assert np.allclose(np.linalg.norm(normal), 1.0)  # Unit length
        assert np.allclose(np.dot(normal, edge), 0)      # Perpendicular to edge
        assert np.dot(normal, center_to_midpoint) > 0    # Points outward


@pytest.mark.parametrize("normals,edges", [
    (
        [np.array([0, 1]), np.array([1, 0]), np.array([0, -1])],  # normals
        [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])]   # edges
    ),
    (
        [np.array([1, 1])/np.sqrt(2), np.array([-1, 1])/np.sqrt(2), np.array([0, -1])],  # normals
        [np.array([1, -1]), np.array([1, 1]), np.array([-2, 0])]   # edges
    )])
def test_scaled_normal_computation(normals, edges):
    """Test computation of scaled normal vectors. Does not test for expected but for charasteristics of scaled normals."""
    scaled_normals = [n * np.linalg.norm(e) for n, e in zip(normals, edges)]
    
    for normal, edge, scaled in zip(normals, edges, scaled_normals):
        # Verify scaling
        assert np.allclose(np.linalg.norm(scaled), np.linalg.norm(edge))
        # Verify direction preserved
        assert np.allclose(scaled/np.linalg.norm(scaled), normal)
        # Verify perpendicularity maintained
        assert np.allclose(np.dot(scaled, edge), 0)


@pytest.mark.parametrize("cell,neighbors,expected_indices", [
    (   # Multiple neighbors
        [0, 1, 2],
        [[1, 2, 3], [3, 1, 2], [4, 5, 0]], [0, 1]
    ),
    (   # No neighbors
        [0, 1, 2],
        [[3, 4, 5], [6, 7, 8]], []
    ),
    (   # All are neighbors
        [0, 1, 2],
        [[1, 2, 3], [2, 0, 4], [0, 1, 5]], [0, 1, 2]
    ),
    (   # Empty neighbor list
        [0, 1, 2],
        [], []
    )])
def test_neighbor_computation(cell, neighbors, expected_indices):
    """Test identification of neighboring triangles based on shared edges."""
    neighbor_indices = [
        idx for idx, neighbor in enumerate(neighbors)
        if len(set(cell) & set(neighbor)) == 2]
    assert neighbor_indices == expected_indices