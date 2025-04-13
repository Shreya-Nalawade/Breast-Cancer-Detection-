from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)

# Load Trained Model
model = tf.keras.models.load_model("breast_cancer_model.h5")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Define Upload Folder
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to Process Image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route for Home & Prediction
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Preprocess & Predict
            image = preprocess_image(file_path)
            prediction = model.predict(image)[0][0]

            # Convert Prediction to Label
            result = "Malignant" if prediction > 0.5 else "Benign"
            confidence = round((prediction if prediction > 0.5 else 1 - prediction) * 100, 2)

            return render_template(
                "index.html", uploaded_image=file_path, result=result, confidence=confidence
            )

    return render_template("index.html", uploaded_image=None)

# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)

#How to Run
'''
1) python app.py
2)run this on browser: http://127.0.0.1:5000
'''