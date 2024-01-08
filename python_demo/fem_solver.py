import numpy as np

def assemble_global_matrix(K_global, K_element, element_indices):
    """
    Assemble the global stiffness matrix from element stiffness matrices.

    :param K_global: Global stiffness matrix.
    :param K_element: Element stiffness matrix.
    :param element_indices: Indices of the vertices forming the element.
    """
    for i, global_i in enumerate(element_indices):
        for j, global_j in enumerate(element_indices):
            K_global[global_i*2:global_i*2+2, global_j*2:global_j*2+2] += K_element[i*2:i*2+2, j*2:j*2+2]
            
# ----------------------------------------------------------------------------------------------------------------------

def solve(vertices, elements, material_model, constraints, forces):
    """
    Solve the FEM problem.

    :param vertices: Array of vertex positions.
    :param elements: Array of elements (vertex indices).
    :param material_model: Material model with a method to compute stiffness matrix.
    :param constraints: Dictionary of vertex constraints {vertex_index: [x_constraint, y_constraint]}.
    :param forces: Numpy array of applied forces at each vertex.
    :return: Numpy array of vertex displacements.
    """
    num_vertices = len(vertices)
    dof = num_vertices * 2  # Degrees of freedom (x and y for each vertex)

    # Initialize global stiffness matrix and force vector
    K_global = np.zeros((dof, dof))
    F_global = np.zeros(dof)

    # Assemble global stiffness matrix and global force vector
    for element in elements:
        K_element = material_model.compute_stiffness_matrix(vertices, element)
        assemble_global_matrix(K_global, K_element, element)

    # Apply forces
    for i, force in enumerate(forces):
        F_global[i*2:i*2+2] = force

    # Apply constraints
    for vertex_index, constraint in constraints.items():
        for i in range(2):
            if constraint[i] is not None:  # Constraint applied
                K_global[vertex_index*2+i, :] = 0
                K_global[vertex_index*2+i, vertex_index*2+i] = 1
                F_global[vertex_index*2+i] = constraint[i]

    # Solve the system of equations
    displacements = np.linalg.solve(K_global, F_global)

    return displacements

# 
# ======================================================================================================================
#

def assemble_global_matrices(vertices, elements, material_model):
    num_vertices = len(vertices)
    dof = num_vertices * 2  # Degrees of freedom

    K_global = np.zeros((dof, dof))
    M_global = np.zeros((dof, dof))

    for element in elements:
        K_element = material_model.compute_stiffness_matrix(vertices, element)
        M_element = material_model.compute_mass_matrix(vertices, element)

        for i, vertex_i in enumerate(element):
            for j, vertex_j in enumerate(element):
                K_global[2*vertex_i:2*vertex_i+2, 2*vertex_j:2*vertex_j+2] += K_element[2*i:2*i+2, 2*j:2*j+2]
                M_global[2*vertex_i:2*vertex_i+2, 2*vertex_j:2*vertex_j+2] += M_element[2*i:2*i+2, 2*j:2*j+2]

    return K_global, M_global

def apply_constraints(K_global, M_global, constraints):
    for vertex_index, constraint in constraints.items():
        for i in range(2):
            if constraint[i] is not None:
                idx = 2 * vertex_index + i
                K_global[idx, :] = 0
                K_global[:, idx] = 0
                K_global[idx, idx] = 1

                M_global[idx, :] = 0
                M_global[:, idx] = 0
                M_global[idx, idx] = 1

def initialize_dynamic_vectors(dof, initial_conditions):
    if initial_conditions is None:
        u = np.zeros(dof)  # Displacement
        v = np.zeros(dof)  # Velocity
        a = np.zeros(dof)  # Acceleration
    else:
        u, v, a = initial_conditions

    return u, v, a

# ----------------------------------------------------------------------------------------------------------------------

def solve_dynamic(vertices, elements, material_model, constraints, forces_func, time_steps, delta_t, initial_conditions, time_integrator):
    """
    Solve the dynamic FEM problem using a specified time integration method.

    :param vertices: Array of vertex positions.
    :param elements: Array of elements (vertex indices).
    :param material_model: Material model with methods to compute stiffness and mass matrices.
    :param constraints: Dictionary of vertex constraints.
    :param forces_func: Function to compute time-dependent forces.
    :param time_steps: Total number of time steps in the simulation.
    :param delta_t: Time step size.
    :param initial_conditions: Initial conditions (initial displacements and velocities).
    :param time_integrator: Function to perform time integration.
    :return: Array of displacements at each time step.
    """
    num_vertices = len(vertices)
    dof = num_vertices * 2  # Degrees of freedom

    # Initialize global matrices and vectors
    K_global, M_global = assemble_global_matrices(vertices, elements, material_model)
    # Apply constraints to global matrices
    apply_constraints(K_global, M_global, constraints)

    # Initialize displacement, velocity, and acceleration vectors
    u, v, a = initialize_dynamic_vectors(dof, initial_conditions)

    displacements_over_time = [u.copy()]

    # Time integration loop
    for step in range(1, time_steps):
        t = step * delta_t
        F = forces_func(t, vertices, dof)  # Compute external forces at time t

        # Use the provided time integrator function
        u, v, a = time_integrator(M_global, K_global, F, u, v, a, delta_t)

        displacements_over_time.append(u.copy())

    return displacements_over_time
