import pandas as pd

# -----------------------------------------------------------------------------
# plotting helper
# -----------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, y: str = "fidelity_proxy") -> None:
    import matplotlib.pyplot as plt
    for circ in sorted(df.circuit.unique()):
        sub = df[df.circuit == circ]
        for combo in sorted(sub.combo.unique()):
            sub2 = sub[sub.combo == combo]
            xs = sub2.depth_p.values
            ys = sub2[y].values
            plt.plot(xs, ys, marker="o", label=f"{circ}-{combo}")
    plt.xlabel("depth p")
    plt.ylabel(y)
    plt.title(f"COMPASS reproduction metric: {y}")
    plt.legend()
    plt.tight_layout()
    plt.show()