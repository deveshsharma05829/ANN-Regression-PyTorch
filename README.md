# ⚡ Power Plant Energy Output Prediction — ANN Regression (PyTorch)

A feedforward Artificial Neural Network built in PyTorch to predict the net hourly 
electrical energy output (PE) of a Combined Cycle Power Plant, based on ambient 
operating conditions.

## 📌 Overview

Power plant output varies with environmental conditions. This project trains a 
neural network to predict energy output (PE) using four input features:

- AT — Ambient Temperature
- V — Exhaust Vacuum
- AP — Ambient Pressure
- RH — Relative Humidity

## 📊 Dataset

Combined Cycle Power Plant Data Set — UCI Machine Learning Repository.
Link: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
9,568 records collected from a power plant over 6 years (2006–2011).

## 🧠 Model Architecture

Input (4 features)
  → Linear(4, 6) → ReLU
  → Linear(6, 6) → ReLU
  → Linear(6, 1)   [output]

- Loss function: MSE Loss
- Optimizer: Adam
- Epochs: 100
- Batch size: 32

## ⚙️ Workflow

1. Data loading and train/test split (80/20)
2. Feature scaling with StandardScaler
3. Conversion to PyTorch tensors and DataLoader
4. Model training with validation tracking each epoch
5. Best model checkpoint saved (best_model.pt) based on lowest validation loss
6. Evaluation on held-out test set

## 📈 Results

| Metric        | Value  |
|---------------|--------|
| Training MSE  | 21.33  |
| Testing MSE   | 19.63  |
| R² Score      | 0.93   |

The model explains ~93% of the variance in energy output on unseen data.





## 🛠️ Tech Stack

- Python
- PyTorch
- scikit-learn
- pandas / NumPy
- Matplotlib

## 🚀 How to Run

git clone https://github.com/deveshsharma05829/power-plant-energy-ann.git
cd power-plant-energy-ann
pip install -r requirements.txt
jupyter notebook ANN_Regression.ipynb

## 📂 Files

- ANN_Regression.ipynb — main notebook
- dataset.csv — training data
- best_model.pt — saved best model weights

