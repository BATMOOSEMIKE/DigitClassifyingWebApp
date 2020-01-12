from flask import Flask, request, render_template, jsonify
#The following are for the AI code
from keras.models import load_model
from PIL import Image, ImageFilter
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

# We used keras 2.2.5 and tensorflow version 1.6


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/predict', methods=['GET','POST'])
def my_form_post():

    model = load_model('digit_classifier.h5')

    #Loads and converts image to MNIST format
    img = load_img('sample9.png', color_mode = "grayscale", target_size=(28,28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img/255.0

    #Makes and prints prediction
    pred = model.predict(img)
    mypred = pred.argmax()
    result = {
        "output": mypred
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == "__main__":
    app.run(debug=True)
