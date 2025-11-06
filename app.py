import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import time

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Sign Language Detection", layout="wide")
st.title("🤟 Real-Time Sign Language Detection")

# Load YOLO model once
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Ensure this file is in the same directory
    model = YOLO(model_path)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# -------------------------------
# Webcam input + detection logic
# -------------------------------
run = st.checkbox("Start Webcam Detection")
FRAME_WINDOW = st.image([])
camera = None

if run:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("🚫 Could not access webcam. Please check your browser permissions.")
    else:
        st.info("🟢 Press Stop Webcam Detection to end session.")
        while run:
            ret, frame = camera.read()
            if not ret:
                st.warning("⚠️ Failed to read frame from camera.")
                break

            # Run YOLO prediction
            results = model.predict(source=frame, conf=0.5, verbose=False)

            # Plot results
            annotated_frame = results[0].plot()

            # Convert BGR → RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Show frame
            FRAME_WINDOW.image(annotated_frame)

            # Stream update delay
            time.sleep(0.05)
else:
    if camera:
        camera.release()
        cv2.destroyAllWindows()
    st.info("👆 Turn on webcam detection to start.")

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    """
    ---
    🧠 **Developed by [Your Name]**  
    Powered by [YOLOv8](https://github.com/ultralytics/ultralytics) + [Streamlit](https://streamlit.io)
    """
)
