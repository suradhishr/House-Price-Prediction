import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor


# Plot settings
plt.style.use('ggplot')
sns.set_palette("husl")


def load_and_explore_data():
    """Step 1: Dataset Loading and Exploration"""

    print("\n--- Step 1: Loading Dataset ---")

    data = fetch_california_housing()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["MedHouseVal"] = data.target

    print("Dataset Shape:", df.shape)

    print("\nDataset Description:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


def clean_data(df):
    """Step 2: Data Cleaning"""

    print("\n--- Step 2: Data Cleaning ---")

    Q1 = df["MedHouseVal"].quantile(0.25)
    Q3 = df["MedHouseVal"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_clean = df[(df["MedHouseVal"] >= lower) & (df["MedHouseVal"] <= upper)]

    print("Rows before cleaning:", len(df))
    print("Rows after cleaning:", len(df_clean))

    return df_clean


def feature_engineering(df):
    """Step 3: Feature Engineering"""

    print("\n--- Step 3: Feature Engineering ---")

    df_eng = df.copy()

    # Avoid divide by zero
    df_eng["RoomsPerPerson"] = df_eng["AveRooms"] / (df_eng["Population"] + 1)

    df_eng["BedroomRatio"] = df_eng["AveBedrms"] / (df_eng["AveRooms"] + 1)

    X = df_eng.drop("MedHouseVal", axis=1)
    y = df_eng["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Step 4: Model Training"""

    print("\n--- Step 4: Model Training ---")

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            eval_metric="rmse"
        )
    }

    results = {}
    predictions = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "RMSE": rmse,
            "R2": r2
        }

        predictions[name] = y_pred

        print(f"{name} -> R2: {r2:.4f}  RMSE: {rmse:.4f}")

    return results, predictions, models


def visualize_results(results, predictions, y_test, models, feature_names):
    """Step 5: Visualization"""

    print("\n--- Step 5: Visualizing Results ---")

    results_df = pd.DataFrame(results).T

    # Model comparison
    results_df["R2"].plot(
        kind="bar",
        title="Model R2 Comparison"
    )

    plt.ylabel("R2 Score")
    plt.show()

    # Best model
    best_model = max(results, key=lambda x: results[x]["R2"])

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions[best_model],
        alpha=0.4
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted ({best_model})")

    plt.show()

    # Feature importance
    importances = models["Random Forest"].feature_importances_

    feat_imp = pd.Series(importances, index=feature_names)

    feat_imp.nlargest(10).plot(
        kind="barh",
        title="Top 10 Important Features"
    )

    plt.show()


# MAIN PROGRAM
if __name__ == "__main__":

    raw_df = load_and_explore_data()

    clean_df = clean_data(raw_df)

    X_train, X_test, y_train, y_test, feature_names = feature_engineering(clean_df)

    metrics, preds, trained_models = train_and_compare_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    visualize_results(
        metrics,
        preds,
        y_test,
        trained_models,
        feature_names
    )

    best_r2 = max(m["R2"] for m in metrics.values())

    print("\nTarget R2 > 0.75 achieved?:", "YES" if best_r2 > 0.75 else "NO")