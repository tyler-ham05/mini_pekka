import ultralytics
import supervision
import torch
import cv2 as cv
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO

model = YOLO('models/best.pt')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Run YOLO prediction on the current frame
    # verbose=False suppresses console output, set to true for debugging
    results = model.predict(source=frame, save=False, imgsz=320, conf=0.35, verbose=False)
    
    # Check if any objects were detected
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        cls_id = int(box.cls[0])           # numeric class index
        conf   = float(box.conf[0])        # confidence
        label  = results[0].names[cls_id]           # human-readable class name
        print(f"{label}: {conf:.2f}")

    # Plots the boxes, labels, and confidence scores over original frame
    annotated_frame = results[0].plot()
    # Display the resulting frame
    
    cv.imshow('results', annotated_frame)
    
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()