# Objective calculation with given Sigma, Lambda, and omega
import numpy as np

def frobenius_objective(Sigma, Lambda, omega):
    n = Sigma.shape[0]
    residual = Sigma - Lambda.T @ Sigma @ Lambda - omega * np.eye(n)
    return np.linalg.norm(residual, 'fro') ** 2


# Example usage and test case
if __name__ == "__main__":
    import numpy as np

    n = 3
    # Known simple case: if Lambda=0 and omega=0, residual = Sigma
    Sigma = np.array([[4, 2, 1],
                      [2, 3, 0],
                      [1, 0, 2]], dtype=float)
    Lambda = np.zeros((n, n))
    omega = 0.0

    obj = frobenius_objective(Sigma, Lambda, omega)
    expected = np.linalg.norm(Sigma, 'fro') ** 2
    print(f"Objective : {obj:.6f}")
    print(f"Expected  : {expected:.6f}")
    print(f"Match     : {np.isclose(obj, expected)}")

if __name__ == "__main__":
    import numpy as np

    n = 3
    Sigma = np.array([[4, 2, 1],
                      [2, 3, 0],
                      [1, 0, 2]], dtype=float)

    alpha = 0.5
    Lambda = alpha * np.eye(n)   # non-zero, but analytically tractable
    omega = 1.0

    obj = frobenius_objective(Sigma, Lambda, omega)

    # Residual = (1 - alpha^2)*Sigma - omega*I
    residual_expected = (1 - alpha**2) * Sigma - omega * np.eye(n)
    expected = np.linalg.norm(residual_expected, 'fro') ** 2

    print(f"Objective : {obj:.6f}")
    print(f"Expected  : {expected:.6f}")
    print(f"Match     : {np.isclose(obj, expected)}")