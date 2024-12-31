import os
import subprocess
import sys
import venv
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Global variable for the virtual environment
VENV_DIR = "ml_app_env"

def run_command(command):
    """Run a system command and handle errors."""
    try:
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}")
        sys.exit(1)

def ensure_pip_updated():
    """Ensure pip is up to date."""
    print("Updating pip...")
    try:
        run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet", "--user"])
        print("Pip updated successfully.")
    except Exception as e:
        print(f"Failed to update pip: {e}")

def install_library(library_name):
    """Install a library with --user and fallback to virtual environment."""
    try:
        print(f"Installing {library_name}...")
        run_command([sys.executable, "-m", "pip", "install", library_name, "--quiet", "--user"])
        print(f"{library_name} installed successfully.")
    except Exception as e:
        print(f"Failed to install {library_name} with --user: {e}")
        if not os.path.exists(VENV_DIR):
            print("Creating virtual environment...")
            create_virtual_environment()
        activate_virtual_environment()
        run_command([os.path.join(VENV_DIR, "Scripts", "python"), "-m", "pip", "install", library_name, "--quiet"])
        print(f"{library_name} installed successfully in virtual environment.")

def create_virtual_environment():
    """Create a virtual environment."""
    venv.create(VENV_DIR, with_pip=True)
    print(f"Virtual environment created at {VENV_DIR}.")

def activate_virtual_environment():
    """Activate the virtual environment."""
    print("Activating virtual environment...")
    os.environ["PATH"] = os.path.join(VENV_DIR, "Scripts") + os.pathsep + os.environ["PATH"]
    sys.executable = os.path.join(VENV_DIR, "Scripts", "python")

# Install required libraries
def install_required_libraries():
    libraries = ["pandas", "numpy", "scikit-learn", "streamlit", "matplotlib", "shap"]
    for lib in libraries:
        install_library(lib)

# Generate a synthetic dataset
def generate_dataset():
    print("Generating dataset...")
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2.5 * X + np.random.randn(100, 1) * 2
    data = pd.DataFrame({"Feature": X.flatten(), "Target": y.flatten()})
    data.to_csv("dataset.csv", index=False)
    print("Dataset saved as 'dataset.csv'.")

# Train models
def train_models():
    print("Training models...")
    data = pd.read_csv("dataset.csv")
    X = data[["Feature"]]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate models
    lr_mse = mean_squared_error(y_test, lr_model.predict(X_test))
    rf_mse = mean_squared_error(y_test, rf_model.predict(X_test))
    print(f"Linear Regression MSE: {lr_mse:.2f}")
    print(f"Random Forest MSE: {rf_mse:.2f}")

    # Save models
    with open("lr_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print("Models saved as 'lr_model.pkl' and 'rf_model.pkl'.")

# Create Streamlit app
def create_streamlit_app():
    """Create the Streamlit app."""
    print("Creating Streamlit app...")
    app_code = """
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load models
with open("lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

st.set_page_config(page_title="Ultimate ML App", layout="wide")
st.title("🚀 Ultimate ML Prediction App")
st.write("Predict target values, compare models, and explore feature importance.")

# Single Prediction
st.header("Single Prediction")
model_choice = st.selectbox("Choose a model:", ["Linear Regression", "Random Forest"])
input_value = st.number_input("Enter a value for prediction:", value=0.0, step=0.1)

if st.button("Predict"):
    model = lr_model if model_choice == "Linear Regression" else rf_model
    prediction = model.predict(np.array(input_value).reshape(-1, 1))
    st.metric("Prediction", f"{prediction[0]:.2f}")
"""
    try:
        with open("app.py", "w", encoding="utf-8") as f:
            f.write(app_code)
        print("Streamlit app created as 'app.py'.")
    except Exception as e:
        print(f"Error creating app.py: {e}")
        sys.exit(1)

# Run the Streamlit app
def run_streamlit_app():
    print("Launching the Streamlit app...")
    os.system("streamlit run app.py")

if __name__ == "__main__":
    ensure_pip_updated()
    install_required_libraries()
    generate_dataset()
    train_models()
    create_streamlit_app()
    run_streamlit_app()
