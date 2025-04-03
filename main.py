from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import uvicorn

# Define constants
MODEL_PATH = "best_rf_model.pkl"
TRAINING_DATA_PATH = "training_data.csv"

# Load model
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Ensure it's in the correct location.")

model = load_model()

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data model
class PredictionRequest(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int
    outcome: int  # User-provided actual outcome (1 = Diabetic, 0 = Non-Diabetic)

    @validator('glucose', 'insulin', 'bmi', 'blood_pressure', 'skin_thickness', 'diabetes_pedigree_function', 'age')
    def check_positive(cls, value):
        if value < 0:
            raise ValueError('Value must be non-negative')
        return value

@app.get("/")
def home():
    return {"message": "Welcome to the Diabetes Prediction API!"}

@app.post("/predict/")
def predict(request: PredictionRequest):
    # Prepare input data
    data = np.array([[request.pregnancies, request.glucose, request.blood_pressure, 
                      request.skin_thickness, request.insulin, request.bmi, 
                      request.diabetes_pedigree_function, request.age]])

    try:
        # Make a prediction
        prediction = model.predict(data)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        # Retrain model with the new data point
        retrain_model(data, request.outcome)

        return {"prediction": result, "message": "Prediction made and model retrained."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def retrain_model(new_data, new_label):
    try:
        # Load existing training data
        if os.path.exists(TRAINING_DATA_PATH):
            data_df = pd.read_csv(TRAINING_DATA_PATH)
        else:
            data_df = pd.DataFrame(columns=["Pregnancies", "Glucose", "BloodPressure", 
                                            "SkinThickness", "Insulin", "BMI", 
                                            "DiabetesPedigreeFunction", "Age", "Outcome"])

        # Convert new data to DataFrame and append
        new_df = pd.DataFrame(new_data, columns=["Pregnancies", "Glucose", "BloodPressure", 
                                                 "SkinThickness", "Insulin", "BMI", 
                                                 "DiabetesPedigreeFunction", "Age"])
        new_df["Outcome"] = new_label
        data_df = pd.concat([data_df, new_df], ignore_index=True)
        data_df.to_csv(TRAINING_DATA_PATH, index=False)

        # Retrain model
        X = data_df.drop(columns=["Outcome"])
        y = data_df["Outcome"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train a new model
        new_model = LogisticRegression()
        new_model.fit(X_scaled, y)

        # Save updated model
        joblib.dump(new_model, MODEL_PATH)

    except Exception as e:
        print(f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
