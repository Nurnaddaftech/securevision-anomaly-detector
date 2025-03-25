import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib  # For saving and loading models
import numpy as np
import subprocess  # For executing system commands

# Define file paths
data_folder = "data"
logs_folder = "logs"
models_folder = "models"
captured_data_file = os.path.join(data_folder, "captured_network_data.csv")
model_file = os.path.join(models_folder, "rf_model.pkl")

# Ensure directories exist
os.makedirs(data_folder, exist_ok=True)
os.makedirs(logs_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# Function to block an IP
def block_ip(ip_address):
    try:
        # Use iptables to block the IP
        subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"], check=True)
        print(f"IP {ip_address} has been blocked successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error blocking IP {ip_address}: {e}")

# Step 1: Capture or Load Network Data
capture_new = input("Do you want to capture new network data? (yes/no): ").strip().lower()
if capture_new == "yes":
    print("Starting packet capture...")
    from scapy.all import sniff, IP
    packet_data = []

    def capture_packet(packet):
        if packet.haslayer(IP):
            packet_info = {
                "ip_src": packet[IP].src,
                "ip_dst": packet[IP].dst,
                "timestamp": datetime.now(),
                "packet_size": len(packet)
            }
            packet_data.append(packet_info)

    sniff(prn=capture_packet, store=False, count=5000)
    df = pd.DataFrame(packet_data)
    df.to_csv(captured_data_file, index=False)
    print(f"Packet capture complete and saved to {captured_data_file}.")
else:
    if os.path.exists(captured_data_file):
        print(f"Loading existing captured data from: {captured_data_file}")
        df = pd.read_csv(captured_data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        print("No existing data found. Exiting.")
        exit()

# Step 2: Feature Engineering with Adaptive Thresholds
print("Enhancing data with DoS-specific features...")
time_window = 1
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Feature Engineering
df['total_requests_per_sec'] = df.groupby(['ip_src', df['timestamp'].dt.floor(f'{time_window}s')])['timestamp'].transform('count')
df['target_frequency'] = df.groupby(['ip_src', 'ip_dst'])['timestamp'].transform('count')
df['interarrival_time'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
df['unique_destinations'] = df.groupby('ip_src')['ip_dst'].transform('nunique')
df['data_flow_size'] = df.groupby('ip_src')['packet_size'].transform('sum')
df['request_ratio'] = df['total_requests_per_sec'] / (df['unique_destinations'] + 1)
df['normalized_size'] = df['data_flow_size'] / (df['total_requests_per_sec'] + 1)

# Dynamic thresholds
REQUEST_RATE_THRESHOLD = df['total_requests_per_sec'].mean() + df['total_requests_per_sec'].std()
TARGET_FREQUENCY_THRESHOLD = df['target_frequency'].mean() + df['target_frequency'].std()
df['possible_dos'] = (df['total_requests_per_sec'] > REQUEST_RATE_THRESHOLD) | \
                     (df['target_frequency'] > TARGET_FREQUENCY_THRESHOLD)

# Step 3: Define Features and Target
X = df[['total_requests_per_sec', 'target_frequency', 'packet_size', 'interarrival_time',
        'unique_destinations', 'data_flow_size', 'request_ratio', 'normalized_size']]
y = df['possible_dos'].astype(int)

# Step 4: Train-Test Split and SMOTE
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("Balancing training data with SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 5: Train or Load Models
print("Training RandomForestClassifier and IsolationForest...")
contamination_ratio = 0.1
if capture_new == "yes" or not os.path.exists(model_file):
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train_resampled, y_train_resampled)
    joblib.dump(rf, model_file)
    print("RandomForestClassifier trained and saved.")
else:
    print("Loading existing RandomForestClassifier...")
    rf = joblib.load(model_file)

iso_forest = IsolationForest(contamination=contamination_ratio, n_estimators=200, random_state=42)
iso_forest.fit(X_train)

# Step 6: Predict and Combine Results
print("\nEvaluating the model...")
rf_preds = rf.predict(X_test)  # Random Forest Predictions
iso_preds = iso_forest.predict(X_test)
iso_preds = [1 if p == -1 else 0 for p in iso_preds]  # Convert IsolationForest -1 to 1 (anomaly)

# Combine Predictions (majority voting logic)
final_preds = [max(rf_p, iso_p) for rf_p, iso_p in zip(rf_preds, iso_preds)]

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds)
recall = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Step 8: Check for Potential DoS IPs and Prompt to Block
potential_dos_ips = df[df['possible_dos']]['ip_src'].unique()
if len(potential_dos_ips) > 0:
    print("\nPotential DoS attack detected from the following IP addresses:")
    for ip in potential_dos_ips:
        print(f" - {ip}")
        block = input(f"Do you want to block IP {ip}? (yes/no): ").strip().lower()
        if block == "yes":
            block_ip(ip)

# Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, final_preds))

# Visualization
plt.figure(figsize=(10, 6))
plt.bar(['Normal', 'Anomaly'], [len(y_test) - sum(final_preds), sum(final_preds)], color=['blue', 'red'])
plt.title("Anomaly Detection Results")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

print("Script completed successfully. Exiting...")
