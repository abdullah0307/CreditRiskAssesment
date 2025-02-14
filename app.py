from flask import Flask, request, jsonify
import pickle
import pandas as pd
import logging

# Initialize Flask app
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Load the model and preprocessing objects
model = pickle.load(open("random_forest_model.pkl", "rb"))

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

with open("Standard_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the expected data types
data_types = {
    "age": "int64",
    "gender": "int32",
    "owns_car": "int32",
    "owns_house": "int32",
    "no_of_children": "float64",
    "net_yearly_income": "float64",
    "no_of_days_employed": "float64",
    "occupation_type": "int32",
    "total_family_members": "float64",
    "migrant_worker": "float64",
    "yearly_debt_payments": "float64",
    "credit_limit": "float64",
    "credit_limit_used(%)": "int64",
    "credit_score": "float64",
    "prev_defaults": "int64",
    "default_in_last_6months": "int64",
}

# Define categorical and numerical features
categorical_features = ["gender", "owns_car", "owns_house", "occupation_type"]
numerical_features = scaler.feature_names_in_

# Define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # try:
        # Get JSON data from the request
        input_data = request.get_json()

        if not input_data:
            return jsonify({"error": "Invalid JSON data"}), 400

        # Convert categorical selections to expected format
        gender_encoded = "F" if input_data["gender"] == "Female" else "M"
        owns_car_encoded = "Y" if input_data["owns_car"] == "Yes" else "N"
        owns_house_encoded = "Y" if input_data["owns_house"] == "Yes" else "N"
        migrant_worker_encoded = 1.0 if input_data["migrant_worker"] == "Yes" else 0.0

        # Create a dictionary of input features
        input_features = {
            "age": input_data["age"],
            "gender": gender_encoded,
            "owns_car": owns_car_encoded,
            "owns_house": owns_house_encoded,
            "no_of_children": float(input_data["no_of_children"]),
            "net_yearly_income": float(input_data["net_yearly_income"]),
            "no_of_days_employed": float(input_data["no_of_days_employed"]),
            "occupation_type": input_data["occupation_type"],
            "total_family_members": float(input_data["total_family_members"]),
            "migrant_worker": migrant_worker_encoded,
            "yearly_debt_payments": float(input_data["yearly_debt_payments"]),
            "credit_limit": float(input_data["credit_limit"]),
            "credit_limit_used(%)": input_data["credit_limit_used"],
            "credit_score": float(input_data["credit_score"]),
            "prev_defaults": input_data["prev_defaults"],
            "default_in_last_6months": input_data["default_in_last_6months"],
        }

        # Convert the input features to a DataFrame
        input_df = pd.DataFrame([input_features])

        # Encode categorical features
        for feature in categorical_features:
            print(feature)
            input_df[feature] = label_encoder[feature].transform(input_df[feature])

        # Set the data types for the DataFrame
        input_df = input_df.astype(data_types)

        # Scale the numerical features
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Make the prediction
        binary_prediction = model.predict(input_df)
        prediction_label = "Yes" if binary_prediction[0] == 1 else "No"

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction_label})

    # except Exception as e:
    #     app.logger.error(f"Error processing request: {e}")
    #     return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)