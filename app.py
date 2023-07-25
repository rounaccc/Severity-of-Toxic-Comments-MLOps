from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from SeverityOfToxicCommentsEndToEnd.pipeline.prediction import PredictionPipeline

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url='/docs')

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training completed successfully")
    except Exception as e:
        return Response("Training failed with error: " + str(e))
    
@app.post("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        text = obj.predict(text)
        return str(text[0])
    except Exception as e:
        raise e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)