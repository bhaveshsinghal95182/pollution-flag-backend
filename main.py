from flask import Flask, request, send_file, jsonify
import tensorflow as tf
import numpy as np
from flask_cors import CORS
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import io
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow all origins

# Load the ML model
model = tf.keras.models.load_model("car_pollution_model.keras")

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "API started"})

@app.route("/predict", methods=["POST"])  # Change to POST
def predict():
    if not request.is_json:
        return jsonify({"error": "Invalid JSON request"}), 400

    data = request.get_json()
    print(f"Received JSON Body: {data}")

    try:
        features = np.array([[data["size"], data["co2"], data["nox"], data["pm25"], data["co"]]])
        prediction = np.argmax(model.predict(features))
        print(prediction)
        return jsonify({"pollution_flag": int(prediction)})
    
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/csvupload", methods=["POST"])
def csvupload():
        # Load the trained model
    model = keras.models.load_model("car_pollution_model.keras")
    
    # Read the uploaded CSV file
    file = request.files['file']
    df = pd.read_csv(file)
    
    # Encode 'Size of Vehicle' column
    size_encoder = LabelEncoder()
    df['Size of Vehicle'] = size_encoder.fit_transform(df['Size of Vehicle'])
    
    # Encode 'Flag' column (target variable)
    flag_encoder = LabelEncoder()
    df['Flag'] = flag_encoder.fit_transform(df['Flag'])
    
    # Select features
    features = ['Size of Vehicle', 'CO2 Emission (g/km)', 'NOx Emission (g/km)', 'PM2.5 Emission (g/km)', 'CO Emission (g/km)']
    X = df[features].values
    
    # Normalize emissions
    scaler = StandardScaler()
    X[:, 1:] = scaler.fit_transform(X[:, 1:])
    
    # Predict
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Convert numeric labels back to categories
    df["Predicted Pollution Level"] = flag_encoder.inverse_transform(predicted_classes)
    
    # Define custom colors for each pollution level
    custom_colors = {"Red": "red", "Yellow": "orange", "Green": "green"}
    
    # Create plot
    plt.figure(figsize=(10, 6))
    for flag in df["Predicted Pollution Level"].unique():
        subset = df[df["Predicted Pollution Level"] == flag]
        plt.scatter(subset["CO2 Emission (g/km)"], subset["NOx Emission (g/km)"], 
                    label=flag, color=custom_colors.get(flag, "gray"))
    
    plt.xlabel("CO2 Emission (g/km)")
    plt.ylabel("NOx Emission (g/km)")
    plt.title("Model Predictions for Pollution Levels")
    plt.legend()
    
    # Save plot to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    plt.close()
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
