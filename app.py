from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Permitir peticiones desde tu HTML (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    user_input = data.get("text", "")

    # Aquí llamas a tu modelo de OpenAI o al tuyo propio
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente científico experto en exoplanetas."},
            {"role": "user", "content": f"Analiza estos datos: {user_input}"}
        ]
    )

    reply = response.choices[0].message.content
    return {"response": reply}

import joblib
model = joblib.load("xgb_exoplanet_model.joblib")

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    features = data.get("features", [])
    prediction = model.predict([features])
    return {"response": f"Predicción del modelo: {int(prediction[0])}"}
