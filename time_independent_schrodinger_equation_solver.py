import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Optional

from model_components import SpatialGrid, StationaryPotential
import constants

from scipy.linalg import eigh


@dataclass
class TiseSolverParameters:
    spatial_grid: SpatialGrid
    stationary_potential: StationaryPotential

    vanishing_boundary_conditions: bool = True
    num_eigenstates: int = 5


class TiseSolver:

    def __init__(self, *args, **kwargs):
        self.parameters = TiseSolverParameters(*args, **kwargs)
        self.eigen_energies = None
        self.eigen_functions = None

    def solve(self):
        hamiltonian = self._construct_hamiltonian(
            vanishing_boundary_conditions=self.parameters.vanishing_boundary_conditions)

        eigen_energies, eigen_functions = eigh(hamiltonian)
        indices = np.argsort(eigen_energies)  # sort eigenvalues

        self.eigen_energies = np.real(eigen_energies[indices])
        self.eigen_functions = eigen_functions[:, indices]

        self.parameters.stationary_potential.eigen_energies = self.eigen_energies[:self.parameters.num_eigenstates]
        self.parameters.stationary_potential.eigen_functions = self.eigen_functions[:self.parameters.num_eigenstates]

    def _construct_hamiltonian(self, vanishing_boundary_conditions: bool = True) -> np.ndarray:
        x_num = self.parameters.spatial_grid.num_grid_points
        x_step = self.parameters.spatial_grid.grid_step

        h_bar = constants.REDUCED_PLANK_CONSTANT
        m_e = constants.ELECTRON_MASS

        diag_matrix = np.diag(np.ones(x_num))
        super_diag_matrix = np.diag(np.ones(x_num - 1), 1)
        sub_diag_matrix = np.diag(np.ones(x_num - 1), -1)

        laplacian = (-2 * diag_matrix + super_diag_matrix + sub_diag_matrix) / (x_step ** 2)

        if vanishing_boundary_conditions:
            laplacian[0, 0] = 0
            laplacian[0, 1] = 0
            laplacian[1, 0] = 0
            laplacian[-1, -1] = 0
            laplacian[-1, -2] = 0
            laplacian[-2, -1] = 0

        hamiltonian = - h_bar ** 2 / (2 * m_e) * laplacian + np.diag(self.parameters.stationary_potential.potential)

        return hamiltonian

