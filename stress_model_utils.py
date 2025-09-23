import os
import torch
import numpy as np
import pandas as pd
import pickle

# Load NN2 assets once, globally
with open(os.path.join(BASE_DIR, "models", "scaler_x_NN2.pkl"), "rb") as f:
    x_scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "scaler_y_NN2.pkl"), "rb") as f:
    y_scaler = pickle.load(f)

model = torch.jit.load(os.path.join(BASE_DIR, "models", "NN2.3_best_model_scripted.pt"))
model.eval()

def load_and_prepare_node_data(csv_path, cut_location):
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = ["Node", "X", "Y", "Z"]

    beam_length = 24.0
    df["Cut"] = cut_location
    df["DistanceFromCut"] = np.abs(df["X"] - df["Cut"])
    df["NormalizedDistanceFromCut"] = df["DistanceFromCut"] / beam_length
    df["RadialDist"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
    df["abs_Y"] = np.abs(df["Y"])
    df["XZ_dist"] = np.sqrt(df["X"]**2 + df["Z"]**2)

    features = df[["X", "Y", "Z", "Cut", "NormalizedDistanceFromCut", "RadialDist", "abs_Y", "XZ_dist"]].values
    features_scaled = x_scaler.transform(features)

    return df, features_scaled

def predict_stress(features_scaled):
    with torch.no_grad():
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        pred_scaled = model(input_tensor).squeeze().cpu().numpy()
        pred_log = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
        stress = np.expm1(pred_log).flatten()
    return stress
