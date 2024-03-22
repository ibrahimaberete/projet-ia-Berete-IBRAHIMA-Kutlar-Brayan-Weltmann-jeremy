from fastapi import FastAPI, HTTPException
import pandas as pd
from typing import List

tags_metadata=[{
  "name":"POST Training",
  "description":"Opération qui permet de recevoir un jeu de données et d'entraîner un modèle sur les données envoyées",
},{
    "name":"POST Predict",
  "description":"Opération qui permet de faire une prédiction à partir du dernier modèle sauvegardé",
},
   {"name": "GET Model", "description": "Opération qui fait appel soit à l’API de OpenAI, soit celle de HuggingFace"},
]
app = FastAPI(
    title="Football Prediction API",
    description="This API allows you to train a model on football match data and make predictions.",
    version="1.0"
)

# Load data from data.csv file
data = pd.read_csv("data.csv")

# Model to store the trained model
trained_model = None

@app.post("/train/")
async def train_model():
    global data
    global trained_model
    if data is None:
        raise HTTPException(status_code=400, detail="Data not loaded.")
    # Your training logic here using data
    # Example: training a simple linear regression model
    from sklearn.linear_model import LinearRegression
    X = data[['home_score', 'away_score']]
    y = data['result']  # Assuming 'result' is a target column
    model = LinearRegression()
    model.fit(X, y)
    trained_model = model
    return {"message": "Model trained successfully."}

@app.post("/predict/")
async def predict(result: List[List[float]]):
    global trained_model
    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")
    # Your prediction logic here using trained_model
    predictions = trained_model.predict(result)
    return {"predictions": predictions}

# Documentation routes
@app.get("/", tags=["Root"], summary="Root endpoint", description="Welcome message for the API.")
async def read_root():
    return {"message": "Welcome to the football prediction API!"}

@app.get("/docs/", tags=["Documentation"], summary="API documentation", description="Documentation for the API.")
async def read_docs():
    return {"message": "API documentation"}

@app.get("/docs/train/", tags=["Documentation"], summary="Training documentation", description="Documentation for training the model.")
async def read_train_docs():
    return {"message": "Documentation for training the model"}

@app.get("/docs/predict/", tags=["Documentation"], summary="Prediction documentation", description="Documentation for making predictions.")
async def read_predict_docs():
    return {"message": "Documentation for making predictions"}

