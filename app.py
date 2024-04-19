from flask import Flask, render_template
import pickle

app = Flask(__name__)

# Load evacuation planning data from the pickle file
with open('evacuation.pkl', 'rb') as file:
    areas_with_special_needs = pickle.load(file)

# Convert state_counts to a dictionary for easier handling in the template
# state_counts_dict = state_counts.to_dict()

# Load the predictions from the pickle file
with open('predicted_disaster.pkl', 'rb') as file:
    predictions = pickle.load(file)

# Access the predictions from the dictionary
total_deaths_pred = predictions['total_deaths_pred']
injured_pred = predictions['injured_pred']
affected_pred = predictions['affected_pred']
total_damage_pred = predictions['total_damage_pred']
reconstruction_costs_pred = predictions['reconstruction_costs_pred']

# count = state_counts['state_counts']


@app.route('/home')
def index():
    # Render the index.html file from the templates folder
    return render_template('home.html')


@app.route('/')
def home():
    # Render the index.html file from the templates folder
    return render_template('index.html',
                           total_deaths_pred=total_deaths_pred,
                           injured_pred=injured_pred,
                           affected_pred=affected_pred,
                           total_damage_pred=total_damage_pred,
                           reconstruction_costs_pred=reconstruction_costs_pred,
                           areas_with_special_needs=areas_with_special_needs)


if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)
