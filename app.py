from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH = 'model/mobilenetv2/mobilenetv2_cleaned.h5'

# Load model
model = load_model(MODEL_PATH, compile=False)

# Classes
class_names = ['Coccidiosis', 'Healthy', 'NewCastleDisease', 'Salmonella']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = model.predict(img_array)
            score = tf.nn.softmax(prediction[0])
            predicted_class = class_names[np.argmax(score)]
            confidence = round(100 * np.max(score), 2)

            return render_template('predict.html',
                                   image_file=filename,
                                   prediction=predicted_class,
                                   confidence=confidence)

    # When method is GET, just render the empty upload form
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
