import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "AI_A2_2527330.csv"
MODEL_PATH = BASE_DIR / "models" / "clustering_model.joblib"
REPORT_PATH = BASE_DIR / "artifacts" / "clustering_report.json"
PLOT_PATH = BASE_DIR / "artifacts" / "plots" / "clusters.png"

SEED = 7330

FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]

VERIFICATION_CODE = "AI-A2-7330"


def main():

    print("=" * 60)
    print("MUSANZE HARVEST INTELLIGENCE SERVICE")
    print("CLUSTERING")
    print("=" * 60)
    print("Student ID:        25/27330")
    print("Verification Code: AI-A2-7330")
    print("Seed:              7330")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()

    # Remove exact duplicate rows before clustering.
    # This prevents the deliberate duplicate rows from
    # artificially influencing cluster sizes.
    X = X.drop_duplicates().reset_index(drop=True)

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    preprocessing = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    X_scaled = preprocessing.fit_transform(X)

    # ---------------------------------------------------------
    # Test K = 2 through K = 6
    # ---------------------------------------------------------

    silhouette_results = {}

    print("\nSilhouette scores:")

    for k in range(2, 7):

        model = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=20,
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
        )

        silhouette_results[str(k)] = float(score)

        print(
            f"K={k}: silhouette = {score:.4f}"
        )

    # ---------------------------------------------------------
    # Select best K
    # ---------------------------------------------------------

    best_k = max(
        silhouette_results,
        key=silhouette_results.get,
    )

    best_k_int = int(best_k)

    print("\nSelected K:", best_k_int)

    # ---------------------------------------------------------
    # Train final clustering model
    # ---------------------------------------------------------

    final_kmeans = KMeans(
        n_clusters=best_k_int,
        random_state=SEED,
        n_init=20,
    )

    labels = final_kmeans.fit_predict(X_scaled)

    # ---------------------------------------------------------
    # Cluster profiles
    # ---------------------------------------------------------

    profile_data = X.copy()
    profile_data["cluster"] = labels

    cluster_profiles = {}

    for cluster_id in sorted(profile_data["cluster"].unique()):

        cluster_data = profile_data[
            profile_data["cluster"] == cluster_id
        ]

        cluster_profiles[str(int(cluster_id))] = {
            "rows": int(len(cluster_data)),
            "rainfall_mean": round(
                float(cluster_data["rainfall_mm"].mean()), 2
            ),
            "temperature_mean": round(
                float(cluster_data["avg_temp_c"].mean()), 2
            ),
            "soil_moisture_mean": round(
                float(cluster_data["soil_moisture_pct"].mean()), 2
            ),
            "fertilizer_mean": round(
                float(cluster_data["fertilizer_kg"].mean()), 2
            ),
            "field_size_mean": round(
                float(cluster_data["field_size_ha"].mean()), 2
            ),
            "cooperative_support_rate": round(
                float(cluster_data["cooperative_support"].mean()), 2
            ),
        }

    # ---------------------------------------------------------
    # Plain-language cluster descriptions
    # ---------------------------------------------------------

    descriptions = {}

    for cluster_id, profile in cluster_profiles.items():

        descriptions[cluster_id] = (
            f"Cluster {cluster_id} contains {profile['rows']} records. "
            f"Average rainfall is {profile['rainfall_mean']} mm, "
            f"average temperature is {profile['temperature_mean']} °C, "
            f"average soil moisture is {profile['soil_moisture_mean']}%, "
            f"average fertilizer use is {profile['fertilizer_mean']} kg, "
            f"and average field size is {profile['field_size_mean']} ha. "
            f"Cooperative support is present in approximately "
            f"{profile['cooperative_support_rate'] * 100:.0f}% "
            f"of records."
        )

    # ---------------------------------------------------------
    # Save complete clustering pipeline
    # ---------------------------------------------------------

    final_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("kmeans", final_kmeans),
        ]
    )

    # Fit the complete pipeline again on the original unique data
    final_pipeline.fit(X)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_pipeline,
        MODEL_PATH,
    )

    # ---------------------------------------------------------
    # Save clustering report
    # ---------------------------------------------------------

    report = {
        "student_id": "25/27330",
        "verification_code": VERIFICATION_CODE,
        "seed": SEED,
        "features": FEATURES,
        "tested_k": [2, 3, 4, 5, 6],
        "silhouette_scores": silhouette_results,
        "selected_k": best_k_int,
        "cluster_profiles": cluster_profiles,
        "cluster_descriptions": descriptions,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
        )

    # ---------------------------------------------------------
    # Create 2D cluster plot
    # ---------------------------------------------------------

    # Use first two standardized features for visualization.
    plot_data = pd.DataFrame(
        {
            "x": X_scaled[:, 0],
            "y": X_scaled[:, 1],
            "cluster": labels,
        }
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for cluster_id in sorted(
        plot_data["cluster"].unique()
    ):

        cluster_points = plot_data[
            plot_data["cluster"] == cluster_id
        ]

        ax.scatter(
            cluster_points["x"],
            cluster_points["y"],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
        )

    ax.set_xlabel(
        "Standardized Rainfall"
    )

    ax.set_ylabel(
        "Standardized Temperature"
    )

    ax.set_title(
        f"Musanze Harvest Clusters "
        f"(K={best_k_int}) | AI-A2-7330"
    )

    ax.legend()

    PLOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_PATH,
        dpi=150,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS")
    print("=" * 60)

    print(
        f"Selected K:       {best_k_int}"
    )

    print(
        f"Best silhouette:  "
        f"{silhouette_results[str(best_k_int)]:.4f}"
    )

    print(
        f"Model saved:      {MODEL_PATH}"
    )

    print(
        f"Report saved:     {REPORT_PATH}"
    )

    print(
        f"Plot saved:       {PLOT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()