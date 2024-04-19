import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load data
df = pd.read_csv('SVI_2020_US.csv')

df1 = df.copy()
df1['NEEDED_SUPPLIES'] = None

# Correlation heatmap
columns_of_interest = ['AREA_SQMI', 'E_TOTPOP', 'M_TOTPOP',
                       'EP_ASIAN', 'EP_AIAN', 'EP_TWOMORE']
subset_df = df[columns_of_interest]
corr_matrix = subset_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Selected Variables')
plt.show()

# Feature selection and linear regression
selected_features = ['AREA_SQMI', 'E_TOTPOP', 'M_TOTPOP',
                     'EP_ASIAN', 'EP_AIAN', 'EP_TWOMORE']
X = df1[selected_features]
y = df1['ST']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predicting new data
new_data = {
    'AREA_SQMI': [4.0],
    'E_TOTPOP': [2500],
    'M_TOTPOP': [500],
    'EP_ASIAN': [1.5],
    'EP_AIAN': [0.2],
    'EP_TWOMORE': [3.0]
}
new_df = pd.DataFrame(new_data)
prediction = model.predict(new_df)
print(f"Predicted amount of needed supplies: {prediction[0]}")

# K-means clustering for identifying areas in need of emergency shelters
X = df[['E_TOTPOP', 'E_HU', 'E_HH', 'E_POV150', 'E_UNEMP',
        'E_NOHSDP', 'E_UNINSUR', 'E_AGE65', 'E_AGE17']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel('Scaled Total Population')
plt.ylabel('Scaled Housing Units')
plt.title('K-means Clustering of Areas')
plt.colorbar(label='Cluster')
plt.show()

# Identify areas in need of emergency shelters
shelter_areas = df[df['cluster'] == 2]
print("Areas in need of emergency shelters:")
print(shelter_areas[['STATE', 'COUNTY', 'LOCATION', 'cluster']])

state_counts = shelter_areas['STATE'].value_counts()
print(shelter_areas)
print("Count of states where emergency shelters are needed:")
print(state_counts)
type(state_counts)


# Create a plan for evacuation, accounting for those with special needs
selected_columns = ['STATE', 'COUNTY', 'LOCATION', 'E_TOTPOP', 'E_HH', 'E_AGE65', 'E_NOVEH', 'E_GROUPQ',
                    'E_LIMENG', 'M_LIMENG', 'EP_LIMENG', 'MP_LIMENG', 'EPL_LIMENG', 'E_AGE17']

evacuation_plan_df = df[selected_columns]

# Define criteria for special needs
special_needs_criteria = (evacuation_plan_df['E_NOVEH'] == 0) | (evacuation_plan_df['E_AGE65'] > 0) | (evacuation_plan_df['E_GROUPQ'] > 0) | (evacuation_plan_df['E_LIMENG'] > 0) | (
    evacuation_plan_df['M_LIMENG'] > 0) | (evacuation_plan_df['EP_LIMENG'] > 0) | (evacuation_plan_df['MP_LIMENG'] > 0) | (evacuation_plan_df['EPL_LIMENG'] > 0) | (evacuation_plan_df['E_AGE17'] > 0)

# Filter areas with special needs
areas_with_special_needs = evacuation_plan_df[special_needs_criteria]
areas_with_special_needs['People_Need_Help'] = areas_with_special_needs['E_TOTPOP']
print("Special need for people", areas_with_special_needs['People_Need_Help'])
# Save areas_with_special_needs to a pickle file named 'evacuation.pkl'
with open('evacuation.pkl', 'wb') as file:
    pickle.dump(areas_with_special_needs, state_counts, file)


print("Evacuation planning data has been saved to 'evacuation.pkl'")
