import cv2
import pytesseract as tess
import numpy as np
from PIL import Image

from PIL import ImageFilter
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

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
        #tess.pytesseract.tesseract_cmd = r"C:\Users\Ahad\AppData\Local\Tesseract-OCR\tesseract.exe"
        img=Image.open(file_path)
        img.filter(ImageFilter.SHARPEN)
        text_frm_img = tess.image_to_string(img)
        print(text_frm_img)
        
       
        

        # Process your result for human   
        return text_frm_img
        
    return None

if __name__ == '__main__':
    app.run(debug=True)






