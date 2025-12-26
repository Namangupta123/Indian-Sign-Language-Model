from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image

app = FastAPI()

MODEL = tf.keras.models.load_model('ISL_model.h5')

LABELS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 
    7: "I", 8: "K", 9: "L", 10: "M", 11: "N", 12: "O", 13: "P", 
    14: "Q", 15: "R", 16: "S", 17: "T", 18: "U", 19: "V", 20: "W", 
    21: "X", 22: "Z"
}

@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (128, 128))
    img_normalized = img_resized / 255.0
    img_final = np.expand_dims(img_normalized, axis=0)

    predictions = MODEL.predict(img_final)
    class_idx = np.argmax(predictions)
    
    predicted_sign = LABELS.get(class_idx, "Unknown")

    return predicted_sign

@app.get("/")
async def health_check():
    return {"status": "running", "model_loaded": MODEL is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)