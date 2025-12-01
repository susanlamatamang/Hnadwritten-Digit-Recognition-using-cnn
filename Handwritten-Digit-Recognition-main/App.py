import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os

# ---------------------------
# Prediction Function
# ---------------------------
def predictDigit(image):
    model = tf.keras.models.load_model("model/digits_recognition_cnn.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    return np.argmax(pred[0]), np.max(pred[0]), img[0].reshape(28, 28)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")

st.markdown("""
    <style>
        body { background-color: #0e1117; color: white; }
        .title { text-align: center; font-size: 42px; font-weight: bold; color: #00ff99; }
        .subtitle { text-align: center; font-size: 18px; color: #aaa; margin-bottom: 20px; }
        .prediction-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            animation: fadeIn 0.5s ease-in-out;
        }
        .history-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin: 5px;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🖍️ Handwritten Digit Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Draw your digit below </div>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
# Initialize History
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # Each entry: (digit, confidence, image_array)

# ---------------------------
# Layout: Canvas + Preview + Controls
# ---------------------------
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("### ✏️ Drawing Canvas")
    stroke_width = st.slider('Brush Size', 5, 30, 15)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        key="canvas"
    )

with col3:
    st.markdown("### ⚙️ Controls")
    if st.button("🚀 Predict Now", use_container_width=True):
        if canvas_result.image_data is not None:
            input_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_array.astype('uint8'), 'RGBA')

            os.makedirs("prediction", exist_ok=True)
            input_image.save('prediction/image.png')

            img = Image.open("prediction/image.png")
            res, conf, model_input_img = predictDigit(img)

            # Save to history (limit last 5 predictions)
            st.session_state.history.insert(0, (res, conf, model_input_img))
            st.session_state.history = st.session_state.history[:5]

            confidence_threshold = 0.5
            if 0 <= res <= 9:
                if conf >= confidence_threshold:
                    st.markdown(f"""
                        <div class="prediction-card" style="border-left: 5px solid #00ff99;">
                            <h2>🎯 Predicted Digit: <b style='color:#00ff99;'>{res}</b></h2>
                            <p style='color: #00ff99;'>Confidence: {conf:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="prediction-card" style="border-left: 5px solid orange;">
                            <h2>⚠️ Low Confidence</h2>
                            <p style='color: orange;'>Confidence: {conf:.2%}</p>
                            <p>Try drawing more clearly.</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ Invalid input. Draw a digit between 0–9.")
        else:
            st.warning("⚠️ Please draw something first.")

# ---------------------------
# Prediction History
# ---------------------------
if st.session_state.history:
    st.write("---")
    st.markdown("## 📜 Prediction History")
    hist_cols = st.columns(len(st.session_state.history))
    for i, (digit, conf, img_array) in enumerate(st.session_state.history):
        with hist_cols[i]:
            st.markdown(f"""
                <div class="history-card">
                    <h4>{digit}</h4>
                    <p style="color: {'#00ff99' if conf >= 0.5 else 'orange'};">{conf:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
            st.image(img_array, width=70, clamp=True)

# ---------------------------
# Sidebar Info
# ---------------------------
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
This interactive app uses a **deep learning CNN model** trained on the MNIST dataset to recognize digits drawn on a canvas.  

**How to use:**  
1. Draw a digit clearly in the black box.  
2. Check the **live preview** to see what the model will process.  
3. Hit **Predict Now** to get results!  

**Tips for best accuracy:**  
- Use the full canvas space  
- Draw with a single stroke if possible  
- Avoid extra marks or shapes  
""")
