import matplotlib.pyplot as plt
import pandas as pd


def save_diagnostic_plots(y_true, y_pred, output_prefix):
    data = pd.DataFrame({"actual": y_true, "predicted": y_pred})

    plt.figure(figsize=(6, 5))
    plt.scatter(data["actual"], data["predicted"], alpha=0.6)
    plt.plot([0, 100], [0, 100], color="gray", linestyle="--")
    plt.xlabel("Actual Risk Score")
    plt.ylabel("Predicted Risk Score")
    plt.title("Prediction Fit")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_fit.png", dpi=150)
    plt.close()

    residuals = data["actual"] - data["predicted"]
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, color="#4C72B0", alpha=0.8)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_residuals.png", dpi=150)
    plt.close()
