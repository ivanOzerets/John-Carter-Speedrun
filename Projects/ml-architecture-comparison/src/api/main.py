from fastapi import FastAPI, UploadFile, File
from src.api.model import load_model, predict

app = FastAPI()
model = load_model("models/transfer_resnet50_gradual_full.pt")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Nature Classifier API"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(model, image_bytes)
    return result