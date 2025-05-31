from ultralytics import YOLO
import cv2
import numpy as np
from fastapi import UploadFile
from tempfile import NamedTemporaryFile


model = YOLO("runs/detect/train/weights/best.pt")

def detect_objects(file: UploadFile):
   
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        contents = file.file.read()
        temp.write(contents)
        temp_path = temp.name

    results = model(temp_path)

  
    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": model.names[class_id],
                "confidence": round(confidence, 2),
                "box": [round(x1), round(y1), round(x2), round(y2)]
            })

    return detections
