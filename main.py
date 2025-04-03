from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import uvicorn

# Define constants
MODEL_PATH = "best_rf_model.pkl"
TRAINING_DATA_PATH = "diabetes_data_upload.csv"

# Load model
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Ensure it's in the correct location.")

model = load_model()

# Initialize label encoders for categorical columns
categorical_columns = [
    'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
    'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity'
]

# Create encoders for categorical features
encoders = {col: LabelEncoder() for col in categorical_columns}

# Load training data to fit encoders (this should ideally be done once and saved)
def fit_encoders():
    if os.path.exists(TRAINING_DATA_PATH):
        train_data = pd.read_csv(TRAINING_DATA_PATH)
        for col in categorical_columns:
            if train_data[col].dtype == 'object':  # Apply encoding only on object type columns
                encoders[col].fit(train_data[col])  # Fit encoder on the training data

fit_encoders()

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
    age: int
    gender: str
    polyuria: str
    polydipsia: str
    sudden_weight_loss: str
    weakness: str
    polyphagia: str
    genital_thrush: str
    visual_blurring: str
    itching: str
    irritability: str
    delayed_healing: str
    partial_paresis: str
    muscle_stiffness: str
    alopecia: str
    obesity: str
    outcome: int  # User-provided actual outcome (1 = Diabetic, 0 = Non-Diabetic)

    @validator('age')
    def check_positive(cls, value):
        if value < 0:
            raise ValueError('Age must be non-negative')
        return value

@app.get("/")
def home():
    return {"message": "Welcome to the Diabetes Prediction API!"}

@app.post("/predict/")
def predict(request: PredictionRequest):
    # Prepare input data
    data = {
        'Age': request.age,
        'Gender': request.gender,
        'Polyuria': request.polyuria,
        'Polydipsia': request.polydipsia,
        'sudden weight loss': request.sudden_weight_loss,
        'weakness': request.weakness,
        'Polyphagia': request.polyphagia,
        'Genital thrush': request.genital_thrush,
        'visual blurring': request.visual_blurring,
        'Itching': request.itching,
        'Irritability': request.irritability,
        'delayed healing': request.delayed_healing,
        'partial paresis': request.partial_paresis,
        'muscle stiffness': request.muscle_stiffness,
        'Alopecia': request.alopecia,
        'Obesity': request.obesity
    }

    # Convert categorical variables to numeric using the trained encoder
    for col in categorical_columns:
        if col in data:
            data[col] = encoders[col].transform([data[col]])[0]  # Transform categorical to numeric

    # Create an array for prediction
    data_array = np.array([[
        data['Age'], data['Gender'], data['Polyuria'], data['Polydipsia'], data['sudden weight loss'],
        data['weakness'], data['Polyphagia'], data['Genital thrush'], data['visual blurring'],
        data['Itching'], data['Irritability'], data['delayed healing'], data['partial paresis'],
        data['muscle stiffness'], data['Alopecia'], data['Obesity']
    ]])

    try:
        # Scale input data using the same scaler (if needed)
        scaler = StandardScaler()  # Assuming this scaler is the one used in training
        data_scaled = scaler.fit_transform(data_array)

        # Make a prediction
        prediction = model.predict(data_scaled)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        # Retrain model with the new data point
        retrain_model(data_array, request.outcome)

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