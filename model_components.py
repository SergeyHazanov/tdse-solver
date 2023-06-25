import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, TypedDict

PotentialTypes = Literal['square', 'coulomb', 'feshbach']

NUM_EIGENSTATE_DENSITY_BINS = 50
BANDWIDTH_ADJUST = 0.15


@dataclass
class SpatialGrid:
    num_grid_points: Optional[int] = None  # specify either num_grid_points or grid_step
    grid_step: Optional[float] = None
    grid_length: float = None  # the grid will be defined between [-grid_length / 2, grid_length / 2]
    grid: np.ndarray = None

    def __post_init__(self):

        if self.num_grid_points and self.grid_step is None:
            # define the spatial grid according to the number of points
            self.grid = np.linspace(-self.grid_length / 2, self.grid_length / 2, self.num_grid_points)
            self.grid_step = self.grid[1] - self.grid[0]

        elif self.num_grid_points is None and self.grid:
            # define the spatial grid according to the grid step
            self.grid = np.arange(-self.grid_length / 2, self.grid_length, self.grid_step)
            self.num_grid_points = len(self.grid)

        else:
            raise Exception('Specify either "num_grid_points" or "grid_step", but not both.')


class PotentialParameters(TypedDict):
    potential_type: PotentialTypes
    args: Dict


@dataclass
class StationaryPotential:
    parameters: PotentialParameters
    spatial_grid: SpatialGrid

    potential: np.ndarray = None

    eigen_energies = None
    eigen_functions = None

    def __post_init__(self):
        self.potential = self._construct_stationary_potential()

    def _construct_stationary_potential(self) -> np.ndarray:
        x_grid = self.spatial_grid.grid

        match self.parameters['potential_type']:

            case 'feshbach':
                a = self.parameters['args'].get('a') or 0.1
                b = self.parameters['args'].get('b') or 0.8

                potential = (x_grid ** 2 / 2 - b) * np.exp(-a * x_grid ** 2)

            case 'coulomb':
                width = self.parameters['args'].get('width') or 1
                height = self.parameters['args'].get('height') or 1

                potential = - 1 / np.sqrt(((1 / width) * x_grid) ** 2 + 1 / height)

            case 'square':
                width = self.parameters['args'].get('width') or 1
                height = self.parameters['args'].get('height') or 1

                potential = height * (np.heaviside(x_grid - width / 2, 0.5) - np.heaviside(x_grid + width / 2, 0.5))

            case _:
                raise ValueError('Invalid potential type.')

        return potential

    def plot_potential(self, num_eigen_energies: int = 5, show: bool = True, **kwargs):

        x_grid = self.spatial_grid.grid

        plot_eigenstate_density = kwargs.get('plot_eigenstate_density', False)
        num_eigenstate_density_bins = kwargs.get('num_eigenstate_density_bins', NUM_EIGENSTATE_DENSITY_BINS)
        density_bandwidth_adjust = kwargs.get('density_bandwidth_adjust', BANDWIDTH_ADJUST)
        figsize = kwargs.get('figsize', (6.4, 4.8))

        if plot_eigenstate_density:
            fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=figsize)
            ax_hist = axs[0]
            ax_potential = axs[1]
        else:
            fig, ax_potential = plt.subplots(1, 1, figsize=figsize)
            ax_hist = None

        ax_potential.plot(x_grid, self.potential, lw=2, label='stationary potential')

        if self.eigen_energies is not None:
            if not plot_eigenstate_density:
                ax_potential.legend()
                color = None
            else:
                color = 'gray'

            for i, energy in enumerate(self.eigen_energies[: num_eigen_energies]):
                energy_line = np.ones_like(x_grid) * energy
                energy_line[self.potential > energy] = None

                ax_potential.plot(x_grid, energy_line, label=fr'$E_{i}$', lw=2, color=color)

        if plot_eigenstate_density:
            # ax_hist.hist(self.eigen_energies[: num_eigen_energies], bins=num_eigenstate_density_bins,
            #              orientation='horizontal', density=True)
            sns.kdeplot(y=self.eigen_energies[: num_eigen_energies], ax=ax_hist, bw_adjust=density_bandwidth_adjust)

            y_min, y_max = ax_potential.get_ylim()
            ax_hist.set_ylim(y_min, y_max)

            ax_hist.set_xlabel('density', fontsize=14)
            ax_hist.set_ylabel('energy', fontsize=14)

        ax_potential.set_xlabel(r'position', fontsize=14)
        ax_potential.set_ylabel(r'energy', fontsize=14)

        if show:
            plt.show()
