import numpy as np

def central_difference_step(M, K, F, u, v, a, delta_t):
    """
    Perform a time step using the Central Difference Method.

    :param M: Mass matrix.
    :param K: Stiffness matrix.
    :param F: External force vector.
    :param u: Displacement vector.
    :param v: Velocity vector.
    :param a: Acceleration vector.
    :param delta_t: Time step size.
    :return: Updated u, v, a.
    """
    # Compute the effective force
    F_eff = F - np.dot(K, u)
    
    # Update acceleration
    a_new = np.linalg.solve(M, F_eff)

    # Update velocity and displacement
    u = u + delta_t * v + 0.5 * delta_t**2 * a
    v = v + 0.5 * delta_t * (a + a_new)

    return u, v, a_new

def newmark_beta_step(M, K, F, u, v, a, delta_t, beta=0.25, gamma=0.5):
    """
    Perform a time step using the Newmark-Beta Method.

    :param M: Mass matrix.
    :param K: Stiffness matrix.
    :param F: External force vector.
    :param u: Displacement vector.
    :param v: Velocity vector.
    :param a: Acceleration vector.
    :param delta_t: Time step size.
    :param beta: Newmark-Beta parameter.
    :param gamma: Newmark-Gamma parameter.
    :return: Updated u, v, a.
    """
    # Compute the effective stiffness matrix
    K_eff = K + gamma / (beta * delta_t) * M
    
    # Compute the effective force
    F_eff = F + np.dot(M, (v * gamma / (beta * delta_t) + a * (gamma / (2 * beta) - 1)))

    # Solve for displacement
    u_new = np.linalg.solve(K_eff, F_eff)

    # Update acceleration and velocity
    a_new = (u_new - u) / (beta * delta_t**2) - v / (beta * delta_t) - a * (0.5 / beta - 1)
    v_new = v + delta_t * (gamma * a_new + (1 - gamma) * a)

    return u_new, v_new, a_new
