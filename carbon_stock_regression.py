import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def random_forest_regression(csv_path): 
    # Load dataset
    df = pd.read_csv(csv_path)

    # Keep numeric columns and siteID for filtering later
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    df_numeric['siteID'] = df['siteID']  # Retain siteID for filtering

    # Define target and features (excluding field-measured variables and siteID from features)
    y = df_numeric['carbon_stock']
    X = df_numeric.drop(columns=[
        'carbon_stock',  # target
        'biomass', 'height', 'maxCrownDiameter', 'ninetyCrownDiameter', 'elevation',
        'stemDiameter',
        # 'latitude', 'longitude',  # field-measured variables
        'siteID', # exclude siteID from features
    ])

    print(f"X columns: {X.columns}")
    print(f"X rows: {X.shape[0]}")

    # Retain siteID for filtering predictions
    site_ids = df_numeric['siteID']

    # Split train, test based on scientificName
    # This ensures that the same species is not present in both train and test sets
    train_indices = []
    test_indices = []
    for taxon, group in df_numeric.groupby(df['taxonID']):
        idx = group.index
        if len(idx) < 2:
            train_indices.extend(idx)
            continue
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    site_train = site_ids.loc[train_indices]
    site_test = site_ids.loc[test_indices]

    # Initialize Random Forest model
    rf = RandomForestRegressor(n_estimators=230, random_state=42, max_depth=20, min_samples_leaf=8, min_samples_split=8)

    rf.fit(X_train, y_train)

    print(type(X_test))

    # Evaluate on the test set
    y_pred_test = rf.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_pmae = test_mae / np.mean(y_test)

    print(f"\nTest Set Results:")
    print(f"Test RMSE: {test_rmse:.2f} kg")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f} kg")
    print(f"Test PMAE: {test_pmae * 100:.4f}%")

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 10))
    sns.barplot(x=importances, y=importances.index, palette='viridis')
    plt.tick_params(labelsize=20)
    plt.title("Feature Importances", fontsize=20)
    plt.xlabel("Importance", fontsize=20)
    plt.ylabel("Feature", fontsize=20)
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    plt.show()

    # Predictions for MLBS and OSBS sites
    # Create a DataFrame with predictions, actual values, and site IDs
    results_df = pd.DataFrame({
        'siteID': site_test,
        'actual_carbon_kg': y_test,
        'predicted_carbon_kg': y_pred_test
    })

    # Filter for MLBS and OSBS sites
    mlbs_results = results_df[results_df['siteID'] == 'MLBS']
    osbs_results = results_df[results_df['siteID'] == 'OSBS']

    # Compute RMSE and R² for MLBS
    mlbs_rmse = np.sqrt(mean_squared_error(mlbs_results['actual_carbon_kg'], mlbs_results['predicted_carbon_kg']))
    mlbs_r2 = r2_score(mlbs_results['actual_carbon_kg'], mlbs_results['predicted_carbon_kg'])
    mlbs_mae = mean_absolute_error(mlbs_results['actual_carbon_kg'], mlbs_results['predicted_carbon_kg'])
    mlbs_pmae = mlbs_mae / np.mean(mlbs_results['actual_carbon_kg'])

    # Compute RMSE and R² for OSBS
    osbs_rmse = np.sqrt(mean_squared_error(osbs_results['actual_carbon_kg'], osbs_results['predicted_carbon_kg']))
    osbs_r2 = r2_score(osbs_results['actual_carbon_kg'], osbs_results['predicted_carbon_kg'])
    osbs_mae = mean_absolute_error(osbs_results['actual_carbon_kg'], osbs_results['predicted_carbon_kg'])
    osbs_pmae = osbs_mae / np.mean(osbs_results['actual_carbon_kg'])

    # Print results for MLBS
    print("\nResults for MLBS Site:")
    print(f"Number of individuals: {len(mlbs_results)}")
    print(f"RMSE: {mlbs_rmse:.2f} kg")
    print(f"R²: {mlbs_r2:.4f}")
    print(f"MAE: {mlbs_mae:.2f} kg")
    print(f"PMAE: {mlbs_pmae * 100:.4f}%")

    # Print results for OSBS
    print("\nResults for OSBS Site:")
    print(f"Number of individuals: {len(osbs_results)}")
    print(f"RMSE: {osbs_rmse:.2f} kg")
    print(f"R²: {osbs_r2:.4f}")
    print(f"MAE: {osbs_mae:.2f} kg")
    print(f"PMAE: {osbs_pmae * 100:.4f}%")

    # Scatter plot for MLBS
    plt.figure(figsize=(8, 8))
    plt.scatter(mlbs_results['actual_carbon_kg'], mlbs_results['predicted_carbon_kg'], alpha=0.6, color='blue')
    plt.plot([mlbs_results['actual_carbon_kg'].min(), mlbs_results['actual_carbon_kg'].max()],
            [mlbs_results['actual_carbon_kg'].min(), mlbs_results['actual_carbon_kg'].max()], 'k--')
    plt.xlabel("Actual carbon (kg)", fontsize=20)
    plt.ylabel("Estimated carbon (kg)", fontsize=20)
    plt.title("MLBS Site: Estimated vs Actual Carbon", fontsize=20)
    plt.grid(True)

    plt.text(
        0.95, 0.10,
        f"RMSE: {mlbs_rmse:.2f} kg\nMAE: {mlbs_mae:.2f} kg\n$R^2$: {mlbs_r2:.4f}",
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=plt.gca().transAxes,
        fontsize=20,
    )

    plt.tight_layout()
    # Save scatter plot for MLBS
    plt.savefig("mlbs_scatter_plot.png")
    plt.show()

    # Scatter plot for OSBS
    plt.figure(figsize=(8, 8))
    plt.scatter(osbs_results['actual_carbon_kg'], osbs_results['predicted_carbon_kg'], alpha=0.6, color='green')
    plt.plot([osbs_results['actual_carbon_kg'].min(), osbs_results['actual_carbon_kg'].max()],
            [osbs_results['actual_carbon_kg'].min(), osbs_results['actual_carbon_kg'].max()], 'k--')
    plt.xlabel("Actual carbon (kg)", fontsize=20)
    plt.ylabel("Estimated carbon (kg)", fontsize=20)
    plt.title("OSBS Site: Estimated vs Actual Carbon", fontsize=20)
    plt.grid(True)

    plt.text(
        0.95, 0.10,
        f"RMSE: {osbs_rmse:.2f} kg\nMAE: {osbs_mae:.2f} kg\n$R^2$: {osbs_r2:.4f}",
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=plt.gca().transAxes,
        fontsize=20,
    )

    plt.tight_layout()
    # Save scatter plot for OSBS
    plt.savefig("osbs_scatter_plot.png")
    plt.show()

    # Save trained model
    model_filename = "carbon_stock_rf_model.pkl"
    joblib.dump(rf, model_filename)
    print(f"Model saved at '{model_filename}'")

def main():
    csv_path = "D:/Self_Practicing/Computer Vision/research/Experiments/datasets/IDTrees_2020/IDTREES_competition_train_v2/train/Field/tree_features_with_biomass_carbon_new_feats.csv"
    
    print("Running Random Forest Regression...")
    random_forest_regression(csv_path)
    
if __name__ == "__main__":
    main()