import cv2
import numpy as np
import tensorflow as tf

# =========================
# Load model
# =========================
model = tf.keras.models.load_model("waste_classifier_model.keras")
class_names = ['mixed', 'organic', 'paper', 'plastic']
IMG_SIZE = 224

# =========================
# AMB82-MINI Stream URL
# =========================
#STREAM_URL = "http://10.194.106.97/video_stream"  # CHANGE THIS

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot connect to AMB82-MINI stream")
    exit()

print("✅ Connected to AMB82-MINI")

frame_count = 0
predicted_class = None
confidence = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Frame not received")
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis=0)

    frame_count += 1

    if frame_count % 10 == 0:
        predictions = model.predict(img_array, verbose=0)
        confidence = np.max(predictions)
        predicted_class = class_names[np.argmax(predictions)]

    if predicted_class is not None and confidence > 0.6:
        label = f"{predicted_class} ({confidence*100:.1f}%)"
    else:
        label = "Detecting..."

    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("AMB82 Waste Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()