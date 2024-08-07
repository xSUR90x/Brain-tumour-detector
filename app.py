from flask import Flask, request, jsonify,render_template,url_for,redirect
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import io
from PIL import Image
import base64

app = Flask(__name__)

# Load your model
model = load_model('bestmodel.keras')

def prepare_image(file):
    img = Image.open(io.BytesIO(file.read()))
    img = img.convert('RGB')
    img = img.resize((224, 224))  # Load and resize the image
    img_array = img_to_array(img)  # Convert the image to an array
    img1 = preprocess_input(img_array)  # Preprocess the image array
    input_arr = np.array([img1])  # Create a batch of images
    return input_arr

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
         print('No file part')
         return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        print('No selected file ')
        return redirect(url_for('index'))
    if file:
        prepared_image = prepare_image(file)
        print('Image Prepared')
        prediction = model.predict(prepared_image)
        print('Prediction:', prediction)
        
    
        if prediction[0][0] < 0.5:
            result = "Your Brain is completely Healthy and you are fine"
        else:
            result = "You have Brain Tumor and you should see Doctor immediately"
        img_base64 = base64.b64encode(file.read()).decode('utf-8')
        return render_template('result.html', prediction=result,img_data=img_base64)
if __name__ == '__main__':
    app.run(debug=True,port=5003)
