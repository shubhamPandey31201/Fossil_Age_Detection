import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

data = pd.read_csv('/Users/nikhilasornapudi/Fossil/fossil_data.csv')

X = data.drop(['Age'], axis=1)
y = data['Age']

# Identifying column types
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('svd', TruncatedSVD(n_components=50)),
                                 ('regressor', LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_pipeline.fit(X_train, y_train)

for dataset, dataset_name, y_true in [(X_train, 'Train', y_train), (X_test, 'Test', y_test)]:
    y_pred = model_pipeline.predict(dataset)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("Model Performance:")
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared:", r2)
    print("-"*20)
