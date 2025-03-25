from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow Cross-Origin Resource Sharing

# Define file paths
data_folder = "data"
models_folder = "models"
captured_data_file = os.path.join(data_folder, "captured_network_data.csv")
model_file = os.path.join(models_folder, "rf_model.pkl")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# Load or initialize the model
if os.path.exists(model_file):
    rf_model = joblib.load(model_file)
else:
    rf_model = RandomForestClassifier()

@app.route('/')
def home():
    return "SecureVision Backend is Running!"

# Endpoint 1: Capture Data
@app.route('/capture_data', methods=['POST'])
def capture_data():
    # Simulate packet data for demonstration purposes
    data = {
        "ip_src": ["192.168.1.1", "192.168.1.2", "192.168.1.3", "192.168.1.4"],
        "ip_dst": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
        "packet_size": [200, 500, 300, 400]
    }
    df = pd.DataFrame(data)
    df.to_csv(captured_data_file, index=False)
    return jsonify({"message": "Data captured successfully!", "data": data})

# Endpoint 2: Train Model
@app.route('/train_model', methods=['POST'])
def train_model():
    df = pd.read_csv(captured_data_file)
    df['possible_dos'] = [0, 1, 0, 1]  # Simulated binary target for demonstration

    # Define features and target
    X = df[['packet_size']]
    y = df['possible_dos']

    # Balance data using SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train RandomForest model
    rf_model.fit(X_resampled, y_resampled)
    joblib.dump(rf_model, model_file)

    return jsonify({"message": "Model trained successfully!"})

# Endpoint 3: Run Predictions
@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv(captured_data_file)
    X = df[['packet_size']]
    predictions = rf_model.predict(X).tolist()
    return jsonify({"predictions": predictions})

# Endpoint 4: Generate Visualizations
@app.route('/visualize', methods=['GET'])
def visualize():
    try:
        print("Visualize endpoint called")
        df = pd.read_csv(captured_data_file)
        print("Data loaded successfully")

        # Add the 'possible_dos' column dynamically
        REQUEST_RATE_THRESHOLD = df['packet_size'].mean() + df['packet_size'].std()  # Example threshold
        df['possible_dos'] = (df['packet_size'] > REQUEST_RATE_THRESHOLD).astype(int)  # Detect anomalies

        # Count normal and anomaly instances
        normal_count = len(df[df['possible_dos'] == 0])
        anomaly_count = len(df[df['possible_dos'] == 1])
        print(f"Normal: {normal_count}, Anomaly: {anomaly_count}")

        # Plot bar chart for normal vs anomaly
        categories = ['Normal', 'Anomaly']
        counts = [normal_count, anomaly_count]
        colors = ['blue', 'red']

        plt.figure(figsize=(10, 6))
        plt.bar(categories, counts, color=colors)
        plt.title("Anomaly Detection Results")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()

        # Save and send the visualization
        plt.savefig("anomaly_detection_results.png")
        print("Visualization saved successfully")
        plt.close()
        return send_file("anomaly_detection_results.png", mimetype='image/png')
    except Exception as e:
        print(f"Error in visualize endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Endpoint 5: Block IP
@app.route('/block_ip', methods=['POST'])
def block_ip():
    data = request.json
    ip_to_block = data.get("ip")
    if not ip_to_block:
        return jsonify({"error": "IP address not provided"}), 400

    try:
        # Simulate blocking an IP (requires sudo permissions in real scenarios)
        print(f"Blocking IP: {ip_to_block}")
        return jsonify({"message": f"IP {ip_to_block} blocked successfully!"})
    except Exception as e:
        print(f"Failed to block IP {ip_to_block}: {e}")
        return jsonify({"error": f"Failed to block IP {ip_to_block}: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
