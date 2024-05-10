import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV

def load_data(data_path):
    return pd.read_csv(data_path)

def preprocess_data(df):
    X = df.drop(['Age'], axis=1)
    y = df['Age']
    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    ct = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('scaler', StandardScaler(), numerical_cols)
    ], remainder='passthrough')
    X = ct.fit_transform(X)

    return X, y

def train_evaluate_validate_svr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }

    # Setup the grid search
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best estimator
    best_svr = grid_search.best_estimator_

    # Evaluate the best model
    y_pred = best_svr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n")
    print("Best Model Performance:")
    print("RMSE:", rmse)
    print("R-squared:", r2)

    return best_svr

# Load and preprocess the data
data_path = '/Users/nikhilasornapudi/Fossil/fossil_data.csv'
fossil_data = load_data(data_path)
X, y = preprocess_data(fossil_data)

# Train and evaluate the model
best_svr_model = train_evaluate_validate_svr(X, y)
