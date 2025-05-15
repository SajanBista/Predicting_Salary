from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import threading
from io import StringIO

# create a FastAPI instance
app = FastAPI()

# global variable to hold uploaded dataset
dataset = None # user will upload and it will stay here

# lock to make sure model training is thread-safe (prevents overlapping updates) using concept of threading
lock = threading.Lock()

# this dictionary stores the best trained model and its information
best_model_info = {
    "model": None,
    "poly_features": None,
    "degree": None,
    "score": None
}


@app.get("/", response_class=HTMLResponse)
async def home():
    """
    homepage route it iscurrently empty. user can later return an HTML form here for upload, train, and prediction. # i also  tried including html 
    """
    return "<h2>Welcome! Please upload your CSV, train the model, and then make predictions.</h2>"


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    route to upload CSV file. reads the uploaded CSV into a pandas DataFrame and stores it in the global variable `dataset`.
    """
    global dataset
    try:
        # Read file content and convert bytes to string
        content = await file.read()
        dataset = pd.read_csv(StringIO(content.decode()))
        return {"message": f"File '{file.filename}' uploaded successfully."}
    except Exception as e:
        return {"error": f"Failed to read CSV: {str(e)}"}


@app.post("/train")
def train_models():
    """
    Trains multiple Polynomial Regression models (degrees 2 to 5).
    Selects and stores the model with the highest R² score.
    """
    global dataset, best_model_info

    if dataset is None:
        return {"error": "Please upload a CSV file first."}

    try:
        # Assuming first column is not useful, and last column is the target (e.g., salary)
        X = dataset.iloc[:, 1:-1].values  # Features (like "Level")
        y = dataset.iloc[:, -1].values   # Target (like "Salary")

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_score = -float('inf')  # Negative infinity to start comparison
        best_model = None
        best_poly = None
        best_degree = None

        # Try polynomial degrees from 2 to 5 and find the best one
        for degree in range(2, 6):
            poly = PolynomialFeatures(degree=degree)
            X_poly_train = poly.fit_transform(X_train)

            model = LinearRegression()
            model.fit(X_poly_train, y_train)

            # Evaluate model performance using R² score on test data
            score = model.score(poly.transform(X_test), y_test)

            # If this model is better, update the best model info
            if score > best_score:
                best_score = score
                best_model = model
                best_poly = poly
                best_degree = degree

        # Store the best model and related info in a thread-safe way
        with lock:
            best_model_info.update({
                "model": best_model,
                "poly_features": best_poly,
                "degree": best_degree,
                "score": best_score
            })

        return {
            "message": "Model trained successfully.",
            "best_degree": best_degree,
            "r2_score": round(best_score, 4)
        }

    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}


@app.post("/predict")
def predict_salary(level: float = Form(...)):
    """
    Predicts salary (or output) based on the input 'level'.
    Uses the best polynomial regression model selected during training.
    """
    try:
        with lock:
            model = best_model_info.get("model")
            poly = best_model_info.get("poly_features")
            degree = best_model_info.get("degree")

            if model is None or poly is None:
                return {"error": "Model is not trained yet. Please train it first."}

            # Make sure the input is in 2D array shape like [[level]]
            new_level = np.array([[level]])

            # Transform the input to polynomial features (same degree as trained model)
            new_poly = poly.transform(new_level)

            # Predict the salary using the model
            prediction = model.predict(new_poly)

            return {
                "input_level": level,
                "predicted_salary": round(float(prediction[0]), 2),
                "model_degree": degree
            }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
