import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(data_path):
    return pd.read_csv(data_path)

def preprocess_data(df):
    """Prepares the data for model training."""
    X = df.drop(['Age'], axis=1)
    y = df['Age']

    # Identifying column types
    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Encode categorical data
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Standardize numerical data
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y

def train_evaluate_validate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Apply PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)  # Transform test set with same PCA model

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_pca, y_train)

    # Evaluate the model on both training and testing sets
    for name, X_set, y_true in zip(["Training", "Testing"], [X_train_pca, X_test_pca], [y_train, y_test]):
        y_pred = rf_model.predict(X_set)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)  # Calculate RMSE
        r2 = r2_score(y_true, y_pred)

        print("Model Performance: ")
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("R-squared:", r2)
        print("-"*20)

    return rf_model

data_path = '/Users/nikhilasornapudi/Fossil/fossil_data.csv'  # Adjusted path to your uploaded file
fossil_data = load_data(data_path)

X, y = preprocess_data(fossil_data)
model = train_evaluate_validate(X, y)
