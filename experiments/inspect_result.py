import numpy as np

# path = "experiments/output/objective_curve_given_sigma.npz"
path = "experiments/output/NSM1e6_objective_curve_sigma_hat_from_given_sigma.npz"

with np.load(path, allow_pickle=True) as data:
    d_m_values = data["d_m_values"]
    objective_values = data["objective_values"]
    masks = data["selected_support_masks"]
    valid = data["selected_support_valid"]

    for d_m, objective, mask, is_valid in zip(
        d_m_values, objective_values, masks, valid
    ):
        print(
            f"\nD_m = {d_m}, optimal value = {objective}, "
            f"valid = {is_valid}"
        )
        print(mask.astype(int))

        # List only recovered off-diagonal directed edges
        edges = [
            (int(i), int(j))
            for i, j in np.argwhere(mask)
            if i != j
        ]
        print("Recovered edges:", edges)

with np.load(path, allow_pickle=True) as data:
    print("Curve type:", data["curve_type"].item())
    Sigma_hat = data["Sigma"]

print("Shape:", Sigma_hat.shape)
print("Sigma_hat:")
print(Sigma_hat)
