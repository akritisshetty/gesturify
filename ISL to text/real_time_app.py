import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("isl_cnn_model.h5")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> label
class_labels = {v: k for k, v in class_indices.items()}

cap = cv2.VideoCapture(0)
text_output = ""
last_gesture = None
cooldown_frames = 20  # frames to wait before accepting same gesture again
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Define ROI
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    img = cv2.resize(roi, (64,64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict gesture
    pred = model.predict(img, verbose=0)
    class_id = np.argmax(pred)
    gesture = class_labels[class_id]

    # Append gesture to text output with cooldown
    if gesture != last_gesture or frame_count > cooldown_frames:
        text_output += gesture
        last_gesture = gesture
        frame_count = 0
    else:
        frame_count += 1

    # Display ROI and recognized text
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    cv2.putText(frame, f'Gesture: {gesture}', (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f'Text: {text_output}', (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("ISL Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()