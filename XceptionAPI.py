#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


max_length = 34
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('model_24.h5')
# Define a flask app
app = Flask(__name__)
def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


def word_for_id(integer, tokenizer):
 for word, index in tokenizer.items():
     if index == integer:
         return word
 return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html)



@app.route('/predict', methods=['GET', 'POST'])
                           
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print(basepath)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        model= Xception(weights='imagenet')
        model_new = Model(model.input, model.layers[-2].output)
        image = preprocess(image) # preprocess the image
        fea_vec = model_new.predict(image) # Get the encoding vector for the image
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])

        description = generate_desc(model, tokenizer, photo, max_length)
        print("\n\n")
        print(description)
        
       
        # Process your result for human   
        return description
        
    return None

if __name__ == '__main__':
    app.run(debug=True)


                           


