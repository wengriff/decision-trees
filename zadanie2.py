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
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
import umap


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
# print all columns
print(df.columns)

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
y

# %%
# Assuming 'df' is your DataFrame with the car data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using 'Prod. year', 'Mileage', and 'Engine volume' as axes
sc = ax.scatter(df['Prod. year'], df['Mileage'], df['Engine volume'], c=df['Price'], cmap='viridis')

# Adding labels
ax.set_xlabel('Production Year')
ax.set_ylabel('Mileage')
ax.set_zlabel('Engine Volume')

# Adding a color bar to represent price
cbar = plt.colorbar(sc)
cbar.set_label('Price')

plt.show()

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
# Initialize the Decision Tree Regressor
dt_reg = DecisionTreeRegressor(random_state=42, max_depth=8)

# %%
# Fit the model to the training data
dt_reg.fit(X_train, y_train)

# %%
# Evaluate the model
y_train_pred = dt_reg.predict(X_train)
y_test_pred = dt_reg.predict(X_test)

# %%
# Compute metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# %%
# Print metrics
print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
print(f"Train R2 Score: {train_r2}, Test R2 Score: {test_r2}")

# %%
# Visualize the Decision Tree (only if the tree is not too large)
plt.figure(figsize=(20,10), dpi=300)
plot_tree(dt_reg, filled=True, feature_names=X_train.columns, max_depth=1)
plt.show()

# %%
residuals = y_train - y_train_pred
plt.scatter(y_train, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (training set)")
plt.show()

# %%
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (test set)")
plt.show()

# %%
# Initialize the Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)

# Fit the model to the training data
rf_reg.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = rf_reg.predict(X_train)
y_test_pred = rf_reg.predict(X_test)

# Compute metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics
print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
print(f"Train R2 Score: {train_r2}, Test R2 Score: {test_r2}")

# Feature Importance
feature_importance = pd.Series(rf_reg.feature_importances_, index=X_train.columns)
feature_importance.nlargest(10).plot(kind='barh')  # Plot top 10 important features
# name the plot and axes
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# %%
residuals = y_train - y_train_pred
plt.scatter(y_train, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (training set)")
plt.show()

# %%
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (test set)")
plt.show()

# %%
# Initialize the Support Vector Regression model
svm_model = SVR(kernel='linear') # parametre vygeneroval copilot, predpokladam, ze su to nejake "defaultne" parametre

# Fit the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_train = svm_model.predict(X_train)
y_pred_test = svm_model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"Train MSE: {mse_train}, Test MSE: {mse_test}")
print(f"Train R2 Score: {r2_train}, Test R2 Score: {r2_test}")


# %%
residuals = y_train - y_train_pred
plt.scatter(y_train, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (training set)")
plt.show()

# %%
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (test set)")
plt.show()

# %%
# Assuming 'df' is your DataFrame
fig = px.scatter_3d(df, 
                    x='Prod. year', 
                    y='Mileage', 
                    z='Engine volume', 
                    color='Price', # Coloring points by Price
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title='3D Scatter Plot of Cars')

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()


# %%
import plotly.express as px

# # Initialize PCA and reduce to 3 dimensions
# pca = PCA(n_components=3)
# X_train_pca = pca.fit_transform(X_train)
# X_valid_pca = pca.transform(X_valid)
# X_test_pca = pca.transform(X_test)

reducer = umap.UMAP(n_components=3)  # Reduce to 3 dimensions
X_train_umap = reducer.fit_transform(X_train)
X_valid_umap = reducer.transform(X_valid)
X_test_umap = reducer.transform(X_test)


# fig = px.scatter_3d(
#     x=X_train_pca[:, 0], 
#     y=X_train_pca[:, 1],
#     z=X_train_pca[:, 2], 
#     color=y_train,
#     labels={'x': 'PC 1', 'y': 'PC 2', 'z': 'PC 3', 'color': 'Price'} 
# )

# fig.show()

import plotly.express as px

fig = px.scatter_3d(
    x=X_train_umap[:, 0], 
    y=X_train_umap[:, 1],
    z=X_train_umap[:, 2], 
    color=y_train,  # Assuming y_train contains the labels or some continuous value
    labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3', 'color': 'Price'} 
)

fig.show()



# %%
correlation_matrix = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the mask
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlation Matrix', size=15)
plt.show()

# %%
# Assuming 'df' is your DataFrame
corr_matrix = df.corr()

# Extract correlations with 'Price'
price_correlations = corr_matrix['Price'].drop('Price')  # Exclude self-correlation

# Sort the correlations
sorted_correlations = price_correlations.sort_values(key=abs, ascending=False)

# Get top 10 positive correlations with 'Price'
top10_positive = sorted_correlations[sorted_correlations > 0].head(10)

# Get top 10 negative correlations with 'Price'
top10_negative = sorted_correlations[sorted_correlations < 0].head(10)

print("Top 10 Positive Correlations with Price:")
print(top10_positive)

print("\nTop 10 Negative Correlations with Price:")
print(top10_negative)


# %%
# Selecting the features with the biggest positive and negative correlations
selected_features = [
    'Prod. year', 'Category_Jeep', 'Fuel type_Diesel', 'Leather interior', 'Mileage'
]

selected_features = [
    'Prod. year', 'Category_Jeep', 'Fuel type_Diesel', 'Leather interior', 'Mileage',
    'Left wheel', 'Engine volume', 'Gear box type_Tiptronic', 'Turbo engine',
    'Gear box type_Manual', 'Category_Hatchback', 'Category_Sedan', 'Fuel type_CNG'
]


# Creating a new DataFrame with the selected features and the target variable 'Price'
new_df = df[selected_features + ['Price']]

# Display the first few rows of the new dataset
# print(new_df.head())

# grab all boolean columns and convert them to int
bool_columns = df.select_dtypes(include='bool').columns

X = new_df.drop(columns=['Price'])
y = new_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=69)

# Standardize data
scaler = StandardScaler()
# !!!!!
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Initialize the Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=256, random_state=42, max_depth=10)

# Fit the model to the training data
rf_reg.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = rf_reg.predict(X_train)
y_test_pred = rf_reg.predict(X_test)

# Compute metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics
print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
print(f"Train R2 Score: {train_r2}, Test R2 Score: {test_r2}")

# Feature Importance
feature_importance = pd.Series(rf_reg.feature_importances_, index=X_train.columns)
feature_importance.nlargest(10).plot(kind='barh')  # Plot top 10 important features
plt.show()


# %%
residuals = y_train - y_train_pred
plt.scatter(y_train, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (training set)")
plt.show()

# %%
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (test set)")
plt.show()

# %%
# Selecting the features with the biggest positive and negative correlations
selected_features = [
    'Prod. year', 'Engine volume', 'Mileage'
]


# Creating a new DataFrame with the selected features and the target variable 'Price'
new_df = df[selected_features + ['Price']]

# Display the first few rows of the new dataset
print(new_df.head())

# grab all boolean columns and convert them to int
bool_columns = df.select_dtypes(include='bool').columns

X = new_df.drop(columns=['Price'])
y = new_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=69)

# Standardize data
scaler = StandardScaler()
# !!!!!
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Initialize the Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=256, random_state=42, max_depth=10)

# Fit the model to the training data
rf_reg.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = rf_reg.predict(X_train)
y_test_pred = rf_reg.predict(X_test)

# Compute metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics
print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
print(f"Train R2 Score: {train_r2}, Test R2 Score: {test_r2}")
plt.show()


# %%
residuals = y_train - y_train_pred
plt.scatter(y_train, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (training set)")
plt.show()

# %%
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (test set)")
plt.show()

# %%
# Initialize PCA with a variance threshold instead of a fixed number of components
variance_threshold = 0.9  # For example, to retain 95% of the variance
pca = PCA(n_components=variance_threshold)

# Fit and transform the data
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)

# Check how many components were chosen to retain the specified variance
print(f"Number of components chosen to retain {variance_threshold*100}% variance: {pca.n_components_}")


# %%
# Assuming you've already performed PCA as discussed in the previous step
# and have your PCA-transformed data in X_train_pca, X_valid_pca, and X_test_pca

# Convert numpy arrays to pandas DataFrames
# Note: The column names here are just placeholders (PC1, PC2, PC3) representing the principal components
X_train_pca_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2', 'PC3'])
X_valid_pca_df = pd.DataFrame(X_valid_pca, columns=['PC1', 'PC2', 'PC3'])
X_test_pca_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2', 'PC3'])

# Initialize the Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=256, random_state=42, max_depth=10)

# Fit the model to the training data (using PCA-transformed data)
rf_reg.fit(X_train_pca_df, y_train)

# Predict on train and test sets (using PCA-transformed data)
y_train_pred = rf_reg.predict(X_train_pca_df)
y_test_pred = rf_reg.predict(X_test_pca_df)

# Compute metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics
print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
print(f"Train R2 Score: {train_r2}, Test R2 Score: {test_r2}")


# %%
residuals = y_train - y_train_pred
plt.scatter(y_train, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (training set)")
plt.show()

# %%
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals)
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot (test set)")
plt.show()

# %%
kmeans = KMeans(n_clusters=3)
numeric_features = df[['Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags']]
binary_features = df[['Leather interior', 'Turbo engine']]
features_for_clustering = pd.concat([numeric_features, binary_features], axis=1)

clusters = kmeans.fit_predict(features_for_clustering)

# Add clusters to dataframe
features_for_clustering['cluster'] = clusters

fig = px.scatter_3d(features_for_clustering, 
                    x='Prod. year', 
                    y='Engine volume', 
                    z='Mileage',
                    color='cluster')
fig.update_layout(width=800, height=800)
fig.show()



