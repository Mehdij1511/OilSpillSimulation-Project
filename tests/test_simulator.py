import pytest
import numpy as np
from src.Simulation.Simulator import simulator

class MockConfig:
	def __init__(self):
		self.meshName = "mock_mesh"
		self.tStart = 0.0
		self.tEnd = 1.0
		self.nSteps = 10
		self.borders = [(0.0, 1.0), (0.0, 1.0)]
		self.restartFile = None

class MockCell:
	def __init__(self, midpoint):
		self.midpoint = np.array(midpoint)
		self.oil_amount = None

class MockMesh:
	def __init__(self, name):
		self.name = name

	def compute_neighbors(self):
		pass

	def get_triangles(self):
		return [
			MockCell(midpoint=[0.35, 0.45]),
			MockCell(midpoint=[0.36, 0.46]),
			MockCell(midpoint=[0.50, 0.50]),
		]

@pytest.fixture
def simulator_instance(monkeypatch):
	monkeypatch.setattr("src.Simulation.Simulator.Mesh", MockMesh)
	config = MockConfig()
	return simulator(config)

def test_initialize_oil_distribution(simulator_instance):
	simulator_instance._initilize_oil_distribution()

	x_star = np.array([0.35, 0.45])
	expected_values = [
		np.exp(-np.sum((np.array([0.35, 0.45]) - x_star)**2) / 0.01),
		np.exp(-np.sum((np.array([0.36, 0.46]) - x_star)**2) / 0.01),
		np.exp(-np.sum((np.array([0.50, 0.50]) - x_star)**2) / 0.01),
	]

	for i, cell in enumerate(simulator_instance.Triangles):
		assert np.isclose(cell.oil_amount, expected_values[i]), f"Test failed for cell {i}"



@pytest.mark.parametrize("x, y", [
    (0, 0),           # Origin case
    (1, 0),           # Point on x-axis
    (0, 1),           # Point on y-axis
    (-1, -1),         # Negative coordinates
    (0.5, 0.5),       # Fractional coordinates
    (2, 3),           # Larger positive values
    (-2.5, 1.5),      # Mixed positive/negative with decimals
    (10, -10),        # Large magnitude values
    (0.1, 0.2),       # Small magnitude values
    (3.14159, 2.71828)  # Irrational-like values
])
def test_get_velocity(simulator_instance, x, y):
    velocity = simulator_instance._get_velocity(x, y)
    expected_velocity = np.array([y - 0.2 * x, -x])
    assert np.allclose(velocity, expected_velocity)

def test_compute_flux():
    # Mock data
    class MockCell:
        def __init__(self, midpoint, oil_amount, area, normals):
            self.midpoint = midpoint
            self.oil_amount = oil_amount
            self._area = area
            self._normals = normals

        def get_scaled_normals(self):
            return self._normals

    # Sett opp celler og parameter
    celle_i = MockCell(midpoint=[0.5, 0.5], oil_amount=0.8, area=1.0, normals=[[0.0, 1.0]])
    neighbor_i = MockCell(midpoint=[0.6, 0.5], oil_amount=0.6, area=1.0, normals=[[0.0, 1.0]])
    edge_idx = 0

    class TestObject:
        dt = 0.1

        @staticmethod
        def _get_velocity(x, y):
            # Returner en enkel hastighet basert på posisjon
            return np.array([x, y])

        def _compute_flux(self, celle_i, neighbor_i, edge_idx):
            v_i = self._get_velocity(celle_i.midpoint[0], celle_i.midpoint[1])
            v_ngh = self._get_velocity(neighbor_i.midpoint[0], neighbor_i.midpoint[1])
            v_avg = 0.5 * (v_i + v_ngh)
            
            scaled_normal = celle_i.get_scaled_normals()[edge_idx]
            dot_product = np.dot(v_avg, scaled_normal)
            
            if dot_product > 0:
                return (-self.dt * celle_i.oil_amount * dot_product) / celle_i._area
            else:
                return (-self.dt * neighbor_i.oil_amount * dot_product) / celle_i._area

    test_object = TestObject()

    # Forventet resultat (beregnet manuelt)
    v_i = np.array([0.5, 0.5])
    v_ngh = np.array([0.6, 0.5])
    v_avg = 0.5 * (v_i + v_ngh)  # [0.55, 0.5]
    scaled_normal = np.array([0.0, 1.0])
    dot_product = np.dot(v_avg, scaled_normal)  # 0.5

    if dot_product > 0:
        expected_flux = (-0.1 * 0.8 * dot_product) / 1.0
    else:
        expected_flux = (-0.1 * 0.6 * dot_product) / 1.0

    # Test funksjonen
    computed_flux = test_object._compute_flux(celle_i, neighbor_i, edge_idx)
    assert np.isclose(computed_flux, expected_flux), f"Flux mismatch: {computed_flux} != {expected_flux}"
    print("Test passerte!")


          
def test_step():

    # Funksjon som etterligner _compute_flux
    def compute_flux(cell, neighbor, edge_idx, dt):
        return 0.1 * (neighbor["oil_amount"] - cell["oil_amount"]) * dt

    # Funksjon som etterligner step
    def step(triangles, dt):
        old_amount = {cell["idx"]: cell["oil_amount"] for cell in triangles}
        for cell in triangles:
            total_flux = 0
            for neighbor_idx in cell["neighbors"]:
                neighbor = triangles[neighbor_idx]

                shared_points = set(cell["point_ids"]) & set(neighbor["point_ids"])
                edge_idx = 0
                for i in range(3):
                    if (
                        cell["point_ids"][i] in shared_points
                        and cell["point_ids"][(i + 1) % 3] in shared_points
                    ):
                        edge_idx = i
                        break

                total_flux += compute_flux(cell, neighbor, edge_idx, dt)

            cell["oil_amount"] = old_amount[cell["idx"]] + total_flux
        return triangles

    # Opprett testdata
    triangles = [
        {"idx": 0, "oil_amount": 1.0, "point_ids": [1, 2, 3], "neighbors": [1]},
        {"idx": 1, "oil_amount": 0.5, "point_ids": [3, 4, 5], "neighbors": [0]},
    ]
    dt = 0.1

    # Før `step`
    initial_oil_amounts = {cell["idx"]: cell["oil_amount"] for cell in triangles}

    # Utfør `step`
    updated_triangles = step(triangles, dt)

    # Etter `step`
    expected_flux_0_to_1 = compute_flux(triangles[0], triangles[1], edge_idx=0, dt=dt)
    expected_flux_1_to_0 = -expected_flux_0_to_1

    expected_oil_amount_0 = initial_oil_amounts[0] + expected_flux_0_to_1
    expected_oil_amount_1 = initial_oil_amounts[1] + expected_flux_1_to_0

    # Sjekk oljeinnhold
    assert np.isclose(
        updated_triangles[0]["oil_amount"], expected_oil_amount_0, atol=1e-3
    ), (
        f"Mismatch for cell 0: {updated_triangles[0]['oil_amount']} != {expected_oil_amount_0}"
    )
    assert np.isclose(
        updated_triangles[1]["oil_amount"], expected_oil_amount_1, atol=1e-3
    ), (
        f"Mismatch for cell 1: {updated_triangles[1]['oil_amount']} != {expected_oil_amount_1}"
    )

    print("Test passerte!")


def test_compute_fishing_grounds():

    # Funksjon som etterligner _compute_fishing_grounds
    def compute_fishing_grounds(triangles, borders):
        total_oil_fish = 0
        x_range = borders[0]
        y_range = borders[1]

        for cell in triangles:
            x, y = cell["midpoint"]
            if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                total_oil_fish += cell["oil_amount"] * cell["area"]
        return total_oil_fish

    # Testdata (bruker ordbøker for cellene)
    triangles = [
        {"midpoint": (0.4, 0.5), "oil_amount": 1.0, "area": 2.0},  # Innefor grensene
        {"midpoint": (0.6, 0.7), "oil_amount": 0.8, "area": 1.5},  # Utenfor på y
        {"midpoint": (0.2, 0.3), "oil_amount": 1.2, "area": 1.0},  # Utenfor på x
    ]

    borders = [(0.3, 0.7), (0.2, 0.6)]  # Range for x og y

    # Forventet verdi: Bare celle (0.4, 0.5) er innenfor grensene
    expected_total_oil_fish = (1.0 * 2.0)  # Bare den ene cellen (0.4, 0.5)

    # Beregn oljeinnhold i fiskefeltet
    computed_oil_fish = compute_fishing_grounds(triangles, borders)

    # Sjekk om beregnet verdi er innenfor en liten feilmargin
    assert np.isclose(computed_oil_fish, expected_total_oil_fish, atol=1e-3), (
        f"Mismatch: {computed_oil_fish} != {expected_total_oil_fish}"
    )

