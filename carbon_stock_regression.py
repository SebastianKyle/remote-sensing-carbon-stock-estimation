import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam  # Import Adam optimizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

import statsmodels.api as sm

def random_forest_regression(csv_path): 
    # Load dataset
    # df = pd.read_csv("D:/Self_Practicing/Computer Vision/research/Experiments/datasets/IDTrees_2020/IDTREES_competition_train_v2/train/Field/tree_features_with_biomass_carbon_no_resize_no_pca.csv")
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

        # -------------- Uncomment để không sử dụng đặc trưng quang phổ ---------------- #
        # 'EVI',
        # 'SAVI', 'MNDVI', 'VOG', 
        # 'GI', 
        # 'MRESRI', 
        # 'Datt', 
        # 'PPR',
        # 'PSRI', 
        # 'SIPI', 
        # 'PRI',
        # 'ACI',
        # 'SL', 
        # 'Mean690-740',
        # 'p_550',
        # 'p_750',

        # -------------- Uncomment để không sử dụng đặc trưng hình thái ---------------- #
        # 'CA',
        # 'CD',
        # 'H', 
        # 'Hmean', 
        # 'Hstd', 
        # 'PH10', 
        # 'PH25', 
        # 'PH50', 
        # 'PH75', 
        # 'PH90', 
        # 'PH95'
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
    plt.tick_params(labelsize=14)
    plt.title("Mức độ quan trọng của các đặc trưng đối với mô hình rừng ngẫu nhiên")
    plt.xlabel("Mức độ quan trọng")
    plt.ylabel("Đặc trưng")
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    plt.show()

    # Scatter plot: prediction vs actual for test dataset
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel("Trữ lượng carbon thực (kg)")
    plt.ylabel("Trữ lượng carbon dự đoán (kg)")
    plt.title("Trữ lượng carbon dự đoán so với trữ lượng carbon thực trên tập dữ liệu kiểm thử")
    plt.grid(True)

    plt.text(
        0.95, 0.10,
        f"RMSE: {test_rmse:.2f} kg\nMAE: {test_mae:.2f} kg\n$R^2$: {test_r2:.4f}",
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=plt.gca().transAxes,
        fontsize=14,
        # bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    plt.tight_layout()
    # Save scatter plot of test set as an image
    plt.savefig("test_scatter_plot.png")
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
    print(f"rMAE: {mlbs_pmae * 100:.4f}%")

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
    plt.xlabel("Trữ lượng carbon thực (kg)")
    plt.ylabel("Trữ lượng carbon dự đoán (kg)")
    plt.title("Khu vực MLBS: trữ lượng carbon dự đoán so với trữ lượng carbon thực")
    plt.grid(True)

    plt.text(
        0.95, 0.10,
        f"RMSE: {mlbs_rmse:.2f} kg\nMAE: {mlbs_mae:.2f} kg\n$R^2$: {mlbs_r2:.4f}",
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=plt.gca().transAxes,
        fontsize=14,
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
    plt.xlabel("Trữ lượng carbon thực (kg)")
    plt.ylabel("Trữ lượng carbon dự đoán (kg)")
    plt.title("Khu vực OSBS: trữ lượng carbon dự đoán so với trữ lượng carbon thực")
    plt.grid(True)

    plt.text(
        0.95, 0.10,
        f"RMSE: {osbs_rmse:.2f} kg\nMAE: {osbs_mae:.2f} kg\n$R^2$: {osbs_r2:.4f}",
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=plt.gca().transAxes,
        fontsize=14,
    )

    plt.tight_layout()
    # Save scatter plot for OSBS
    plt.savefig("osbs_scatter_plot.png")
    plt.show()

    # Save trained model
    model_filename = "carbon_stock_rf_model.pkl"
    joblib.dump(rf, model_filename)
    print(f"Model saved at '{model_filename}'")

def deep_learning_regression(csv_path):
    # Load dataset
    # df = pd.read_csv("D:/Self_Practicing/Computer Vision/research/Experiments/datasets/IDTrees_2020/IDTREES_competition_train_v2/train/Field/tree_features_with_biomass_carbon_no_resize_no_pca.csv")
    df = pd.read_csv(csv_path)

    # Keep numeric columns and siteID for filtering later
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    df_numeric['siteID'] = df.loc[df_numeric.index, 'siteID']  # Retain siteID for filtering, ensuring index alignment

    # Define target and features (excluding field-measured variables and siteID from features)
    y = df_numeric['carbon_stock']
    X = df_numeric.drop(columns=[
        'carbon_stock',  # target
        'biomass', 'height', 'maxCrownDiameter', 'ninetyCrownDiameter', 'elevation',
        'stemDiameter',
        # 'latitude', 'longitude',  # field-measured variables (if you want to exclude them)
        'siteID', # exclude siteID from features
    ])

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    # Drop X na rows in y
    y = y[X.index]
    df_numeric = df_numeric.loc[X.index]

    print(f"X columns: {X.columns}")
    print(f"X rows: {X.shape[0]}")

    # Separate features into modalities
    phs_features = ['CD', 'CA', 'H', 'Hmean', 'Hstd', 'PH10', 'PH25', 'PH50', 'PH75', 'PH90', 'PH95']
    hps_features = ['EVI', 'SAVI', 'GI', 'Mean690-740', 'MNDVI', 'VOG', 'MRESRI', 'Datt', 'PPR', 'PSRI', 'SIPI', 'PRI', 'ACI', 'SL', 'p_550', 'p_750']

    X_phs = X[phs_features]
    X_hps = X[hps_features]

    scaler_phs = StandardScaler()
    X_phs_scaled = scaler_phs.fit_transform(X_phs)
    scaler_hps = StandardScaler()
    X_hps_scaled = scaler_hps.fit_transform(X_hps)

    train_indices = []
    test_indices = []
    for taxon, group in df_numeric.groupby(df_numeric['taxonID'] if 'taxonID' in df_numeric.columns else df['taxonID']):
        idx = group.index
        if len(idx) < 2:
            train_indices.extend(idx)
            continue
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    # Convert DataFrame indices to positional indices for numpy arrays
    index_to_pos = {idx: pos for pos, idx in enumerate(X.index)}
    train_pos = [index_to_pos[i] for i in train_indices if i in index_to_pos]
    test_pos = [index_to_pos[i] for i in test_indices if i in index_to_pos]

    X_phs_train_scaled = X_phs_scaled[train_pos]
    X_hps_train_scaled = X_hps_scaled[train_pos]
    X_phs_test_scaled = X_phs_scaled[test_pos]
    X_hps_test_scaled = X_hps_scaled[test_pos]

    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    # Define input shapes
    input_shape_phs = (X_phs_train_scaled.shape[1],)
    input_shape_hps = (X_hps_train_scaled.shape[1],)

    # Define input layers for each modality
    input_phs = Input(shape=input_shape_phs, name='rgb_input')
    input_hps = Input(shape=input_shape_hps, name='chm_input')

    embedding_dim = 32

    # Linear projection layers
    projected_phs = Dense(11 * embedding_dim, activation='relu', use_bias=False)(input_phs)
    projected_hps = Dense(18 * embedding_dim, activation='relu', use_bias=False)(input_hps)

    # Concatenate the projected features
    x = concatenate([projected_phs, projected_hps])
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.2)(x)

    # Output layer for regression
    output_layer = Dense(1, activation='linear')(x)

    # Create the model
    model = Model(inputs=[input_phs, input_hps], outputs=output_layer)

    # Compile the model
    # Using a smaller learning rate and gradient clipping to prevent NaNs
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.0002,  # Starting learning rate
        first_decay_steps=1000,     # Number of steps for the first decay
        t_mul=2.0,                 # Factor to multiply the decay steps after each restart
        m_mul=1.0,                 # Factor to multiply the learning rate after each restart
        alpha=0.0001                  # Minimum learning rate
    )
    optimizer = Adam(learning_rate=lr_schedule) # Start with a reasonable learning rate and clipping
    model.compile(optimizer=optimizer, loss='mse') # Using Mean Squared Error for regression

    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    # Train the model
    history = model.fit(
        [X_phs_train_scaled, X_hps_train_scaled], y_train,
        epochs=200,  # Increased epochs slightly
        batch_size=12, # Adjust batch size as needed
        validation_split=0.1, # Use a validation split for monitoring
        callbacks=[early_stop],
        verbose=1 # Show training progress
    )

    # Evaluate the model on the test set
    print("\nEvaluating on Test Set:")
    loss = model.evaluate([X_phs_test_scaled, X_hps_test_scaled], y_test, verbose=0)
    print(f"Test Loss (MSE): {loss}")

    # Predict on the test set
    y_pred_test_dl = model.predict([X_phs_test_scaled, X_hps_test_scaled])

    # Calculate and print metrics for the test set
    test_rmse_dl = np.sqrt(mean_squared_error(y_test, y_pred_test_dl))
    test_r2_dl = r2_score(y_test, y_pred_test_dl)
    test_mae_dl = mean_absolute_error(y_test, y_pred_test_dl)
    test_pmae_dl = (test_mae_dl / np.mean(y_test)) * 100

    print(f"\nDeep Learning Test Set Results:")
    print(f"Test RMSE: {test_rmse_dl:.2f} kg")
    print(f"Test R²: {test_r2_dl:.4f}")
    print(f"Test MAE: {test_mae_dl:.2f} kg")
    print(f"Test PMAE: {test_pmae_dl:.2f}%")

    # Scatter plot: prediction vs actual for the test set
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_test_dl, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel("Actual Carbon (kg)")
    plt.ylabel("Predicted Carbon (kg)")
    plt.title("Deep Learning: Predicted vs Actual Carbon (Test Set)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dl_test_scatter_plot.png")
    plt.show()

    # Save model
    model_filename_dl = "carbon_stock_dl_model.h5"
    model.save(model_filename_dl)
    print(f"Deep Learning model saved at '{model_filename_dl}'")

def linear_regression(csv_path):
    # Load CSV
    # df = pd.read_csv("D:/Self_Practicing/Computer Vision/research/Experiments/datasets/IDTrees_2020/IDTREES_competition_train_v2/train/Field/tree_features_with_biomass_carbon_no_resize_no_pca.csv")
    df = pd.read_csv(csv_path)

    # Keep numeric columns and drop rows with NaNs or Infs
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Features and target
    log_df = np.log(df_numeric + 1e-6)
    X = log_df.drop(columns=['carbon_stock', 
                            'biomass', 'height', 'maxCrownDiameter', 'ninetyCrownDiameter', 'elevation',
                            'stemDiameter', 
                            #  'latitude', 'longitude'
                            'MNDVI' # negative values -> NaN when get log
                            ])
    y = log_df['carbon_stock']

    # Print out columns that contain NaN or Inf values
    nan_columns = X.columns[X.isna().any()].tolist()
    inf_columns = X.columns[(X == np.inf).any() | (X == -np.inf).any()].tolist()
    if nan_columns:
        print(f"Columns with NaN values: {nan_columns}")
    if inf_columns:
        print(f"Columns with Inf values: {inf_columns}")
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]  
    print(f"Valid rows: {len(X)} out of {len(df)} total rows")

    # Split data
    train_indices = []
    test_indices = []
    for taxon, group in df_numeric.groupby(df_numeric['taxonID'] if 'taxonID' in df_numeric.columns else df['taxonID']):
        idx = group.index
        if len(idx) < 2:
            train_indices.extend(idx)
            continue
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    # Convert DataFrame indices to positional indices for numpy arrays
    index_to_pos = {idx: pos for pos, idx in enumerate(X.index)}
    train_pos = [index_to_pos[i] for i in train_indices if i in index_to_pos]
    test_pos = [index_to_pos[i] for i in test_indices if i in index_to_pos]
    X_train = X.iloc[train_pos]
    X_test = X.iloc[test_pos]
    y_train = y.iloc[train_pos]
    y_test = y.iloc[test_pos]

    # Bidirectional stepwise regression
    def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
        included = list(initial_list)
        while True:
            changed = False

            # Forward step
            excluded = list(set(X.columns) - set(included))
            new_pval = pd.Series(index=excluded, dtype=float)
            for new_col in excluded:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_col]]))).fit()
                new_pval[new_col] = model.pvalues[new_col]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f'Add  {best_feature:<30} with p-value {best_pval:.6f}')

            # Backward step
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f'Drop {worst_feature:<30} with p-value {worst_pval:.6f}')

            if not changed:
                break
        return included

    # Run stepwise
    selected_features = stepwise_selection(X_train, y_train)

    # Final model
    X_train_selected = sm.add_constant(X_train[selected_features])
    X_test_selected = sm.add_constant(X_test[selected_features])
    final_model = sm.OLS(y_train, X_train_selected).fit()

    # Evaluation
    y_pred = final_model.predict(X_test_selected)
    # Convert back to normal scale (kg)
    y_test_actual = np.exp(y_test)
    y_pred_actual = np.exp(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    pmae = (mae / np.mean(y_test_actual)) * 100

    print("\nSelected Features:", selected_features)
    print("\nModel Summary:\n", final_model.summary())
    print(f"\nTest RMSE: {rmse:.2f} kg")
    print(f"Test MAE: {mae:.2f} kg")
    print(f"Test R²: {r2:.4f}")
    print(f"Test PMAE: {pmae:.2f}%")

def main():
    csv_path = "D:/Self_Practicing/Computer Vision/research/Experiments/datasets/IDTrees_2020/IDTREES_competition_train_v2/train/Field/tree_features_with_biomass_carbon_no_resize_no_pca.csv"
    
    print("Running Random Forest Regression...")
    random_forest_regression(csv_path)
    
    print("\nRunning Deep Learning Regression...")
    deep_learning_regression(csv_path)
    
    print("\nRunning Linear Regression...")
    linear_regression(csv_path)
    
if __name__ == "__main__":
    main()