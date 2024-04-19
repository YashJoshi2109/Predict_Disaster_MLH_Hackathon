import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pickle

# Load the data from the Excel file
data = pd.read_excel('public_emdat_2024-03-31.xlsx')

# Perform initial data cleaning: drop unnecessary columns, convert data types
data.drop(data.columns[0:2], axis=1, inplace=True)
data = data.convert_dtypes()

# Fill missing numeric values with 0
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(0)

# Define features and target variables
features = ['Magnitude', 'Total Deaths', 'No. Injured', 'No. Affected',
            'Total Damage (\'000 US$)', 'Reconstruction Costs (\'000 US$)', 'Latitude', 'Longitude']

# Define target variables as a list
target_variables = ['Total Deaths', 'No. Injured', 'No. Affected',
                    'Total Damage (\'000 US$)', 'Reconstruction Costs (\'000 US$)']

# Filter the data for specific disaster types (e.g., 'Drought', 'Flood', 'Road')
disaster_types = ['Drought', 'Flood', 'Road']
filtered_data = data[data['Disaster Type'].isin(disaster_types)]

# Group the filtered data by 'Disaster Type' and 'Region'
grouped_data = filtered_data.groupby(['Disaster Type', 'Region'])

# Initialize dictionaries to store total predictions for each target variable
total_deaths_pred = {}
injured_pred = {}
affected_pred = {}
total_damage_pred = {}
reconstruction_costs_pred = {}

# Loop through each group to train models and make predictions
for (disaster_type, region), group_data in grouped_data:
    group_name = f"{disaster_type} - {region}"

    # Separate features and target variables for the current group
    X_group = group_data[features]
    y_group = group_data[target_variables]

    # Calculate test size based on the number of samples in this group
    test_size = 0.2 if len(group_data) >= 5 else 0

    # Split data into training and testing sets for the current group
    if test_size > 0:
        X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(
            X_group, y_group, test_size=test_size, random_state=42)

        # Create Random Forest Regressor models for each target variable
        models = {
            'Total Deaths': RandomForestRegressor(random_state=42),
            'No. Injured': RandomForestRegressor(random_state=42),
            'No. Affected': RandomForestRegressor(random_state=42),
            'Total Damage (\'000 US$)': RandomForestRegressor(random_state=42),
            'Reconstruction Costs (\'000 US$)': RandomForestRegressor(random_state=42),
        }

        # Train each model with the appropriate target variable
        predictions = {}
        for target_var, model in models.items():
            model.fit(X_train_group, y_train_group[target_var])
            y_pred = model.predict(X_test_group)
            predictions[target_var] = int(sum(y_pred))

        # Store the predictions in dictionaries
        total_deaths_pred[group_name] = predictions['Total Deaths']
        injured_pred[group_name] = predictions['No. Injured']
        affected_pred[group_name] = predictions['No. Affected']
        total_damage_pred[group_name] = predictions['Total Damage (\'000 US$)']
        reconstruction_costs_pred[group_name] = predictions[
            'Reconstruction Costs (\'000 US$)']

# Combine all the prediction dictionaries into a single dictionary
all_predictions = {
    'total_deaths_pred': total_deaths_pred,
    'injured_pred': injured_pred,
    'affected_pred': affected_pred,
    'total_damage_pred': total_damage_pred,
    'reconstruction_costs_pred': reconstruction_costs_pred
}

# Save the predictions to a pickle file
with open('predicted_disaster.pkl', 'wb') as file:
    pickle.dump(all_predictions, file)

print("Predictions have been saved to 'predicted_disaster.pkl'")
