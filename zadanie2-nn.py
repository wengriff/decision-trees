# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# %%
# Allow printing more columns
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# %%
# Do not show pandas warnings
pd.set_option('mode.chained_assignment', None)

# %%
df = pd.read_csv('./zadanie2_dataset.csv')

# %%
# Print the number of rows and columns
print("Number of rows and columns: ", df.shape)

# %%
df = df.drop(columns=['ID'])
df = df.drop(columns=['Levy'])
df = df.drop(columns=['Manufacturer'])
df = df.drop(columns=['Model'])

# %%
df = df[(df['Cylinders'] > 3) & (df['Cylinders'] < 13)]
df = df[(df['Price'] >= 500) & (df['Price'] <= 100000)]
df = df[(df['Prod. year'] >= 1992)]
df = df[(df['Engine volume'] >= 1) & (df['Engine volume'] <= 10)]
df['Mileage'] = df['Mileage'].str.replace(' km', '').astype(int)
df = df[(df['Mileage'] >= 0) & (df['Mileage'] <= 500000)]

# %%
# Convert boolean columns 'Turbo engine' and 'Left wheel' to int
df['Turbo engine'] = df['Turbo engine'].astype(int)
df['Left wheel'] = df['Left wheel'].astype(int)
df['Leather interior'] = df['Leather interior'].map({'Yes': 1, 'No': 0})

# %%
df = pd.get_dummies(df, columns=['Color'])
df = pd.get_dummies(df, columns=['Category'])
df = pd.get_dummies(df, columns=['Fuel type'])
df = pd.get_dummies(df, columns=['Gear box type'])
df = pd.get_dummies(df, columns=['Drive wheels'])
df = pd.get_dummies(df, columns=['Doors'])

# %%
# remove duplicate rows
df = df.drop_duplicates()

# %%
# print the number of rows and columns
print("Number of rows and columns: ", df.shape)

# %%
# count the number of null values in each column
print(df.isnull().sum())

# %%
# print all columns
print(df.columns)

# %%
X = df.drop(columns=['Price'])
y = df['Price']

# %%
X

# %%
y

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=69)

# %%
# Standardize data
scaler = StandardScaler()
# !!!!!
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# %%
# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# %%
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlp = MLPRegressor(hidden_layer_sizes=(256, 256, 256, 256, 256), activation='relu', solver='adam', random_state=42, early_stopping=True, n_iter_no_change=10, learning_rate='adaptive')

mlp.fit(X_train, y_train)

y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)


# %%
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"Train MSE: {mse_train}, Test MSE: {mse_test}")
print(f"Train R2 Score: {r2_train}, Test R2 Score: {r2_test}")


# %%
residuals = y_train - y_pred_train
plt.scatter(y_train, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (training set)")
plt.show()

# %%
residuals = y_test - y_pred_test
plt.scatter(y_test, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (test set)")
plt.show()


