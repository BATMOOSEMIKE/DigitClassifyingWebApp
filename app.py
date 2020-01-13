from flask import Flask, request, render_template, request
#The following are for the AI code
from keras.models import load_model
from PIL import Image, ImageFilter
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import keras
from keras import backend as K
import os
import tensorflow as tf

# We used keras 2.2.5 and tensorflow version 1.6


app = Flask(__name__, template_folder='templates')

def init():
   global model,graph
   model = load_model('digit_classifier.h5')
   model._make_predict_function()
   graph = tf.get_default_graph()

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict', methods=['GET','POST'])
def my_form_post():
    if request.method == 'POST':
      img = Image.open(request.files['file'].stream).convert("L")
      img = img.resize((28,28))
      img = img_to_array(img)
      img = img.reshape(1, 28, 28, 1)
      img = img.astype('float32')
      img = img/255.0

      #Makes and prints prediction
      pred = model.predict(img)
      mypred = pred.argmax()

      return 'Predicted Number: ' + str(mypred)

if __name__ == '__main__':
   print(("* Loading Keras model and Flask starting server..."
      "please wait until server has fully started"))
   init()
   app.run(debug = True)
