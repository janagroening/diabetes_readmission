# Goal: Deploy the final model using flask

# dependencies
from joblib import load
from pathlib import Path
from flask import Flask, request, jsonify
import pandas as pd

# paths
PATH_REPO = Path(__file__).parent.parent
PATH_MODELS = PATH_REPO / "models"

# load model
model = load(PATH_MODELS / "final_model.joblib")

# create flask app
app = Flask("predict")

# API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # get data from request
    data = request.get_json()
    
    # convert data to format model expects
    input_data = pd.DataFrame([data])  # Convert dict to DataFrame with one row
    
    # make prediction
    probability = model.predict_proba(input_data)
    prediction = model.predict(input_data)
    
    # return prediction
    return jsonify({
        "predicted_label": prediction.tolist(),
        "predicted_outcome": "readmitted" if prediction == 1 else "not_readmitted",
        "predicted_probability": probability.tolist(),
    })

# run app
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=9696
    )