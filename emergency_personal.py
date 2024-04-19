import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("SVI_2020_US.csv")

# Drop unnecessary columns
data.drop(columns=['STATE', 'ST_ABBR', 'COUNTY', 'LOCATION'], inplace=True)

# Define dependent and independent variables
dependent_var = 'ST'  # Total population as a proxy for required emergency personnel
independent_vars = [
    'AREA_SQMI', 'E_HU', 'E_HH', 'E_POV150', 'E_UNEMP',
    'E_AGE65', 'E_AGE17', 'E_DISABL', 'E_UNINSUR', 'E_MINRTY', 'E_MUNIT',
    'E_MOBILE', 'EP_POV150', 'EP_UNEMP', 'EP_HBURD', 'EP_AGE65', 'EP_AGE17',
    'EP_DISABL', 'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE',
    'E_DAYPOP'
]

# Split the data into training and testing sets
X = data[independent_vars]
y = data[dependent_var]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Initialize the Random Forest Regressor
clf = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model
clf.fit(X_train, y_train)

# Save the trained model to a pickle file named "emergency.pkl"
with open('emergency.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Predict the number of emergency personnel required for the first row of the test set
emergency_personnel_required = clf.predict(X_test.iloc[0:1])
print(
    f"Estimated number of emergency personnel required: {emergency_personnel_required[0]}")

# To load the model from the pickle file (optional):
# with open('emergency.pkl', 'rb') as file:
#     loaded_clf = pickle.load(file)
