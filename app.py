
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model, Model
from keras.utils.vis_utils import plot_model
from keras.preprocessing import image
import keras

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
import wget
from google_drive_downloader import GoogleDriveDownloader as gd
import gdown as gdown


# numpy utils
import numpy as np
app = Flask(__name__)

import os


#model_file = "InceptionResNetV2.h5"

#gdown.download("https://drive.google.com/uc?id=1FSITXcRHK3CknIFE6Do4uE-es_huHT54", output= model_file)



img_width = 224
img_height = 224
preprocess_input_mode = "tf"  #"tf" "torch" "caffe"


#model = keras.applications.inception_resnet_v2.InceptionResNetV2(weights = "imagenet") # tf
#model = keras.applications.resnet50.ResNet50(weights = "imagenet") # caffe
#model = keras.applications.xception.Xception(weights = "imagenet") # tf
model = keras.applications.mobilenet.MobileNet(weights = "imagenet")


#model.save("InceptionResNetV2.h5")

#model = load_model(model_file)

#print(model.summary())




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(img_width, img_height))  # target_size=(224, 224)

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode=preprocess_input_mode) # "tf"

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string

        os.remove(file_path)

        #return result



        return jsonify(
           prediction_class = str(pred_class[0][0][1]),
           prediction_score = str(pred_class[0][0][2])
        )


    return None




if __name__ == '__main__':
    app.run(port=5002, debug=True, threaded=False)
    # Serve the app with gevent

