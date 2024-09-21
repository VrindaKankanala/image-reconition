
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('C:/Users/RITHIKA/rithika/FinalModel/trained_model.h5')

# Set the upload folder path
UPLOAD_FOLDER = 'C:/Users/RITHIKA/rithika/FinalModel/uploads'
PRODUCTS_FOLDER = 'C:/Users/RITHIKA/rithika/FinalModel/Dataset'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PRODUCTS_FOLDER'] = PRODUCTS_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize pixel values
    return image

def get_product_images(predicted_class):
    class_map = {0: 'Emily', 1: 'Rani', 2: 'Anthony', 3: 'Damon'}
    class_folder = os.path.join(app.config['PRODUCTS_FOLDER'], class_map[predicted_class])
    if os.path.exists(class_folder):
        return [os.path.join(class_map[predicted_class], f) for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
    return []

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Preprocess the image
            image = preprocess_image(file_path)
            if image is None:
                return jsonify({'error': 'Unable to process the image.'}), 400
            # Make prediction
            image = np.expand_dims(image, axis=0)
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            product_images = get_product_images(predicted_class)
            return jsonify({'predicted_class': int(predicted_class), 'product_images': product_images})

    return render_template('index.html')

@app.route('/products/<path:filename>')
def serve_product_image(filename):
    return send_from_directory(app.config['PRODUCTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)


