from fastapi import FastAPI, UploadFile, File
from detect import detect_objects

app = FastAPI()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    results = detect_objects(file)
    return {"detections": results}
