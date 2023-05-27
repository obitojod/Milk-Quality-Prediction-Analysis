from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('milky.html')

@app.route('/result', methods=['POST'])
def predict():
    # Get the input values from the HTML form
    ph = float(request.form['ph'])
    temperature = float(request.form['temperature'])
    taste = float(request.form['taste'])
    odor = float(request.form['odor'])
    fat = float(request.form['fat'])
    turbidity = float(request.form['turbidity'])
    color = float(request.form['color'])

    # Create a numpy array with the input values
    input_data = np.array([[ph, temperature, taste, odor, fat, turbidity, color]])

    # Make the prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Convert the prediction to a quality label


    return render_template('milky.html',name=' Quality of the Milk: {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True,port=5000)
