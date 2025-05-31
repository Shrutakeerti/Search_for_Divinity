import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile

model = YOLO("runs/detect/train/weights/best.pt")

st.title("Blueprint Door & Window Detection")

st.write("Upload a blueprint image (PNG/JPG), and the model will detect doors and windows.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
   
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    
    results = model(temp_path)

    
    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            
            x = round(x1)
            y = round(y1)
            w = round(x2 - x1)
            h = round(y2 - y1)

            detections.append({
                "label": model.names[class_id].lower(),  
                "confidence": round(confidence, 2),
                "bbox": [x, y, w, h]
            })

    st.write("### Detections:")
    st.json({"detections": detections})

    
    import cv2

    img_cv = np.array(image)
    for det in detections:
        x, y, w, h = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_cv, f"{label} {conf}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(img_cv, caption="Image with bounding boxes", use_column_width=True)
