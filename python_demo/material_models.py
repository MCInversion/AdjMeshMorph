import numpy as np

class LinearElastic:
    def __init__(self, E, nu, density=1.0):
        """
        Initialize the Linear Elastic material model.

        :param E: Young's modulus
        :param nu: Poisson's ratio
        :param density: Material density
        """
        self.E = E
        self.nu = nu
        self.density = density

    def compute_stiffness_matrix(self, vertices, element):
        """
        Compute the stiffness matrix for a triangular element.

        :param vertices: Array of vertex positions.
        :param element: Indices of the vertices forming the element.
        :return: Element stiffness matrix.
        """
        # Extract vertices of the element
        v1, v2, v3 = vertices[element]

        # Compute area of the triangle
        area = self._triangle_area(v1, v2, v3)

        # Plane stress stiffness matrix
        D = self.E / (1 - self.nu**2) * np.array([[1, self.nu, 0],
                                                  [self.nu, 1, 0],
                                                  [0, 0, (1 - self.nu) / 2]])

        # B matrix (strain-displacement matrix)
        B = self._compute_B_matrix(v1, v2, v3)

        # Element stiffness matrix
        K = area * np.dot(np.dot(B.T, D), B)

        return K

    def _triangle_area(self, v1, v2, v3):
        """
        Compute the area of a triangle.

        :param v1, v2, v3: Vertices of the triangle.
        :return: Area of the triangle.
        """
        return 0.5 * np.linalg.det(np.array([[v1[0], v1[1], 1],
                                             [v2[0], v2[1], 1],
                                             [v3[0], v3[1], 1]]))
        
    def compute_mass_matrix(self, vertices, element):
        """
        Compute the mass matrix for a triangular element.

        :param vertices: Array of vertex positions.
        :param element: Indices of the vertices forming the element.
        :return: Element mass matrix.
        """
        v1, v2, v3 = vertices[element]
        area = self._triangle_area(v1, v2, v3)

        # Assuming consistent mass matrix (simplified)
        # For more accuracy, a lumped mass matrix or other formulations can be used
        mass_matrix = np.zeros((6, 6))
        mass_per_node = self.density * area / 3

        for i in range(3):
            mass_matrix[2*i:2*i+2, 2*i:2*i+2] = mass_per_node * np.eye(2)

        return mass_matrix

    def _compute_B_matrix(self, v1, v2, v3):
        """
        Compute the B matrix (strain-displacement matrix) for a triangle.

        :param v1, v2, v3: Vertices of the triangle.
        :return: B matrix.
        """
        # Inverse of the matrix for derivatives
        M_inv = np.linalg.inv(np.array([[v1[0], v1[1], 1],
                                        [v2[0], v2[1], 1],
                                        [v3[0], v3[1], 1]]))

        # Derivatives of shape functions
        dN = np.array([[M_inv[0, 0], M_inv[1, 0], M_inv[2, 0]],
                       [M_inv[0, 1], M_inv[1, 1], M_inv[2, 1]]])

        # Assemble B matrix
        B = np.zeros((3, 6))
        B[0, 0:6:2] = dN[0, :]
        B[1, 1:6:2] = dN[1, :]
        B[2, 0:6:2] = dN[1, :]
        B[2, 1:6:2] = dN[0, :]

        return B


class NeoHookean:
    def __init__(self, E, nu):
        # Initialize properties
        pass

    def compute_stiffness_matrix(self, element):
        # Compute and return stiffness matrix
        pass
