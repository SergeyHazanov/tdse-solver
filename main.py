from model_components import SpatialGrid, StationaryPotential, PotentialParameters
from time_independent_schrodinger_equation_solver import TiseSolver

if __name__ == '__main__':

    spatial_grid = SpatialGrid(num_grid_points=200, grid_length=50)

    # potential_parameters = PotentialParameters(potential_type='coulomb', args={'height': 1, 'width': 1})
    potential_parameters = PotentialParameters(potential_type='feshbach',
                                               args={'a': 0.1, 'b': 0.8})

    stationary_potential = StationaryPotential(
        parameters=potential_parameters,
        spatial_grid=spatial_grid
    )

    tise_solver = TiseSolver(spatial_grid=spatial_grid, stationary_potential=stationary_potential, num_eigenstates=100,
                             vanishing_boundary_conditions=True)
    tise_solver.solve()

    stationary_potential.plot_potential(num_eigen_energies=30,
                                        show=True,
                                        plot_eigenstate_density=True,
                                        density_bandwidth_adjust=0.3,
                                        num_eigenstate_density_bins=10,
                                        figsize=(15, 10))

