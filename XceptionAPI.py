#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from keras.models import Model
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from flask_cors import CORS, cross_origin

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

# cors = CORS(app, resources={r"/api/http://localhost:4200": {"origins": "http://localhost:4200"}})# api = Api(app)
global wordtoix
global ixtoword
global model1



wordtoix=load(open("wordtoix.p","rb"))
ixtoword=load(open("ixtoword.p","rb"))
model1=load_model('model_24.h5')


# Define a flask app

def generate_desc(model, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
                           
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print('\n\n\n\n',f,'\n\n\n\n')
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print(basepath)
        print('\n\n\n\n',basepath,'\n\n\n\n')
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        # print(file_path)
        
        print('\n\n\n\n file path ',file_path,'\n\n\n\n')
        f.save(file_path)
        max_length = 34
        model= Xception(weights='imagenet')
        model_new = Model(model.input, model.layers[-2].output)
        
        
        img=Image.open(file_path)# Convert PIL image to numpy array of 3-dimensions
        img=img.resize((299,299))
        x = image.img_to_array(img)
        # Add one more dimension
        x = np.expand_dims(x, axis=0)
        # preprocess the images using preprocess_input() from inception module
        x = preprocess_input(x)
        fea_vec = model_new.predict(x) # Get the encoding vector for the image
        fea_vec = fea_vec.reshape((1,2048))
        
        description = generate_desc(model1, fea_vec, max_length)
        print("\n\n")
        print(description)



        
        
        
       
        # Process your result for human   
        return description
        
    return None

if __name__ == '__main__':
    app.run(debug=True)


                           



                           


