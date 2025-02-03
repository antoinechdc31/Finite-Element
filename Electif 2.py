import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve

# Domain parameters
L = 1.0  # Length of the domain
N = 10   # Number of nodes
nodes = np.linspace(0, L, N)

# Define coefficients a, b, and source term f(x)
a = 1.0
b = 1.0

def f(x):
    return np.sin(np.pi * x)  # Example source term

# Basis functions (cubic splines)
def phi(i, x, nodes):
    """Cubic spline basis function."""
    h = nodes[1] - nodes[0]  # Uniform spacing
    if i == 0 or i == len(nodes) - 1:
        return 0  # Boundary nodes
    elif nodes[i - 1] <= x <= nodes[i]:
        return (x - nodes[i - 1]) ** 2 * (3 * nodes[i] - x) / h**3
    elif nodes[i] <= x <= nodes[i + 1]:
        return (nodes[i + 1] - x) ** 2 * (3 * x - nodes[i]) / h**3
    else:
        return 0

# Derivatives of the basis functions
def dphi(i, x, nodes):
    """First derivative of cubic spline basis function."""
    h = nodes[1] - nodes[0]
    if i == 0 or i == len(nodes) - 1:
        return 0
    elif nodes[i - 1] <= x <= nodes[i]:
        return 6 * (x - nodes[i - 1]) / h**3 - 3 / h**2
    elif nodes[i] <= x <= nodes[i + 1]:
        return -6 * (nodes[i + 1] - x) / h**3 + 3 / h**2
    else:
        return 0

def d2phi(i, x, nodes):
    """Second derivative of cubic spline basis function."""
    h = nodes[1] - nodes[0]
    if i == 0 or i == len(nodes) - 1:
        return 0
    elif nodes[i - 1] <= x <= nodes[i]:
        return 6 / h**3
    elif nodes[i] <= x <= nodes[i + 1]:
        return -6 / h**3
    else:
        return 0

# Assemble the stiffness matrix A and load vector F
def assemble_system(nodes):
    N = len(nodes)
    A = np.zeros((N, N))
    F = np.zeros(N)

    for i in range(N):
        for j in range(N):
            # Compute A[i, j]
            integrand = lambda x: d2phi(j, x, nodes) * d2phi(i, x, nodes) + \
                                  a * dphi(j, x, nodes) * dphi(i, x, nodes) + \
                                  b * phi(j, x, nodes) * phi(i, x, nodes)
            A[i, j] = quad(integrand, 0, L)[0]

        # Compute F[i]
        integrand_f = lambda x: f(x) * phi(i, x, nodes)
        F[i] = quad(integrand_f, 0, L)[0]

    # Apply boundary conditions
    A[0, :] = 0
    A[:, 0] = 0
    A[-1, :] = 0
    A[:, -1] = 0
    A[0, 0] = 1
    A[-1, -1] = 1
    F[0] = 0
    F[-1] = 0

    return A, F

# Solve the system
A, F = assemble_system(nodes)
U = solve(A, F)

# Output results
print("Stiffness Matrix A:\n", A)
print("Load Vector F:\n", F)
print("Solution U:\n", U)
