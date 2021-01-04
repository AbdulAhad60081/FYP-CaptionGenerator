import os
import cv2
import pytesseract as tess
import numpy as np
from PIL import Image

from PIL import ImageFilter
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_cors import CORS, cross_origin
from flask import make_response, url_for,Blueprint

# Define a flask app
app = Flask(__name__)

ocr = Blueprint("ocr",__name__)
CORS(ocr, supports_credentials=True)

@ocr.route('/ocr2', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@ocr.route('/predict2', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        print('\n\n\n\n',f,'\n\n\n\n')
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        # print(basepath)
        print('\n\n\n\n basepath ',basepath,'\n\n\n\n')
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        # print(file_path)
        print('\n\n\n\n',file_path,'\n\n\n\n')
        f.save(file_path)
        tess.pytesseract.tesseract_cmd = r"tesseract.exe"
        img=Image.open(file_path)
        img.filter(ImageFilter.SHARPEN)
        text_frm_img = tess.image_to_string(img)
        print(text_frm_img)
        
        print('\n\n\n\n text_frm_img',text_frm_img,'\n\n\n\n')
        
       
        

        # Process your result for human   
        return text_frm_img
        
    return None


