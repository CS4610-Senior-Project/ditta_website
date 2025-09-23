import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os
from stress_model_utils import load_and_prepare_node_data, predict_stress


app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Load the trained Keras model
model = load_model(os.path.join("models", "NN1_trained_model.keras"))

# Load the input scaler
with open(os.path.join("models", "NN1_scaler.pkl"), "rb") as f:
    scaler_X = pickle.load(f)

@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/run_ml_ajax', methods=['POST'])
def run_ml_ajax():
    try:
        # Extract 7 numeric inputs from form
        inputs = [float(request.form[f'input{i}']) for i in range(1, 8)]
        inputs = np.array(inputs).reshape(1, -1)

        # Scale and predict
        inputs_scaled = scaler_X.transform(inputs)
        y_pred = model.predict(inputs_scaled).flatten()

        # Render the fragment and return to be injected into the page
        return render_template('result_fragment.html', y_pred=y_pred[0])

    except Exception as e:
        print(f"Error during AJAX prediction: {e}")
        return "An error occurred during prediction", 500

@app.route('/get_stress_field', methods=['POST'])
def get_stress_field():
    try:
        data = request.get_json()
        cut_location = float(data["cut_location"])

        # Step 1: Load + preprocess the node data
        df, features_scaled = load_and_prepare_node_data(
            os.path.join("data", "Node_Cords_40k.csv"),
            cut_location
        )


        # Step 2: Predict stress
        stress = predict_stress(features_scaled)

        # Step 3: Package results
        coords = df[["X", "Y", "Z"]].values
        output = [{"x": float(x), "y": float(y), "z": float(z), "stress": float(s)} for (x, y, z), s in zip(coords, stress)]
        return jsonify(output)

    except Exception as e:
        print("Error in stress field generation:", e)
        return jsonify({"error": str(e)}), 500

# Do not include on Render / Github 
# for local use only
# if __name__ == "__main__":
    # app.run(debug=True)