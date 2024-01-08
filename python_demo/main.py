import numpy as np
import math
import material_models
import fem_solver
import visualization
import material_parameters as mp
import argparse
import time_integration as ti

def save_displacements_to_file(displacements_over_time, filename="displacements.csv"):
    """
    Save the solution displacements to a CSV file, compatible with both static and dynamic simulations.

    :param displacements_over_time: Displacement values (single array or list of arrays for each time step).
    :param filename: Name of the file to save the displacements.
    """
    if isinstance(displacements_over_time, list):
        # Dynamic simulation with multiple time steps
        num_vertices = len(displacements_over_time[0]) // 2
        num_time_steps = len(displacements_over_time)
        data_to_save = np.zeros((num_vertices, num_time_steps * 2))

        for t in range(num_time_steps):
            reshaped_displacements = displacements_over_time[t].reshape(num_vertices, 2)
            data_to_save[:, 2*t:2*t+2] = reshaped_displacements

        headers = ['X Displacement t={0}, Y Displacement t={0}'.format(t) for t in range(num_time_steps)]
    else:
        # Static simulation with a single time step
        num_vertices = len(displacements_over_time) // 2
        data_to_save = displacements_over_time.reshape(num_vertices, 2)
        headers = ['X Displacement, Y Displacement']

    # Define header for CSV file
    header_string = ','.join(headers)

    # Save to CSV file
    np.savetxt(filename, data_to_save, delimiter=',', header=header_string, comments='')

    print(f"Displacements saved to {filename}")

    
# 
# ======================================================================================================================
# python main.py static
# ----------------------------------------------------------------------------------------------------------------------

def static_simulation():
    # Define mesh data
    vertices = np.array([[0, 0], [1, 0], [2, 0], [0.5, math.sqrt(3)/2], [1.5, math.sqrt(3)/2]])
    elements = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4]])

    # Define material properties
    material_properties = mp.load_material_properties()
    E, nu = mp.get_material_properties("Rubber", material_properties)

    # Define material model
    material_model = material_models.LinearElastic(E, nu)

    # Define boundary conditions and forces
    constraints = {0: [0, 0]}  # Fix vertex 0 in both x and y directions
    forces = np.zeros((len(vertices), 2))  # Initialize force array
    
    # Example: Apply a force to vertex 2
    force_magnitude = 50000  # Example value for force magnitude
    forces[2] = [force_magnitude, 0]  # Modify as needed

    # Perform FEM simulation
    displacements = fem_solver.solve(vertices, elements, material_model, constraints, forces)
    
    # Save displacements to a file
    save_displacements_to_file(displacements, filename="static_displacements.csv")

    # Visualize results
    visualization.plot_mesh(vertices, elements, displacements, scale=1)  # Scale factor for visualization
 
# 
# ======================================================================================================================
# python main.py dynamic
# ----------------------------------------------------------------------------------------------------------------------
   
def dynamic_simulation():
    # Define mesh data
    vertices = np.array([[0, 0], [1, 0], [2, 0], [0.5, math.sqrt(3)/2], [1.5, math.sqrt(3)/2]])
    elements = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4]])

    # Load material properties and define material model
    material_properties = mp.load_material_properties()
    E, nu = mp.get_material_properties("Rubber", material_properties)
    material_model = material_models.LinearElastic(E, nu, density=1000)  # Density is needed for dynamic analysis

    # Define boundary conditions
    constraints = {0: [0, 0]}  # Fix vertex 0 in both x and y directions

    # Define the time-dependent force function
    force_magnitude = 1000  # Example value for force magnitude
    def forces_func(t, vertices, dof):
        forces = np.zeros((dof,))
        forces[4:6] = [force_magnitude * (1 + math.sin(t)), 0]  # Apply dynamic force to vertex 2
        return forces

    # Simulation parameters
    time_steps = 100
    delta_t = 0.5
    initial_conditions = (np.zeros(len(vertices) * 2), np.zeros(len(vertices) * 2), np.zeros(len(vertices) * 2))

    # Perform dynamic FEM simulation
    displacements_over_time = fem_solver.solve_dynamic(
        vertices, elements, material_model, constraints, forces_func, 
        time_steps, delta_t, initial_conditions, ti.newmark_beta_step
    )
    
    # Save displacements to a file
    save_displacements_to_file(displacements_over_time, filename="dynamic_displacements.csv")
    
    # Visualize results
    visualization.animate_mesh(vertices, elements, displacements_over_time, scale=50, 
                               interval=time_steps, delta_t=delta_t)

# 
# ======================================================================================================================
#

def main():
    parser = argparse.ArgumentParser(description="Run FEM simulations.")
    parser.add_argument("simulation_type", choices=['static', 'dynamic'], 
                        help="Type of simulation to run: 'static' or 'dynamic'")
    args = parser.parse_args()

    if args.simulation_type == 'static':
        static_simulation()
    elif args.simulation_type == 'dynamic':
        dynamic_simulation()

if __name__ == "__main__":
    main()
