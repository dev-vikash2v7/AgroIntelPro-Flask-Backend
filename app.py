import os  , json

import joblib
from flask import Flask, request ,jsonify

from crop_disease_prediction import get_pred_img
from crop_recommend import get_crop
from ferti_recommend import get_fertilizer


from flask_cors import CORS


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'User_Upload_Image'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def root():
    return 'hello from AgroIntel Pro'



@app.route("/api/disease_predict" ,  methods=['POST'])
async def disease_predict(): 
    if request.method == 'POST':

        try:         

            if 'image' not in request.files :
                return jsonify({'error': 'No file part'}), 400

            image_file =  request.files['image']
            crop_name =  request.form.get('crop_name')

            if image_file.name == '':
                return  jsonify({'error': 'No selected file'}), 400


            disease_name = get_pred_img( crop_name , image_file)
            # print(disease_name)
            
            return {'diseaseName' : disease_name}


        except Exception as e:
            print('errererere , ' , e) 
            return jsonify({'error': str(e)}), 500




@app.route("/api/crop_recommend" , methods=['POST'])
def crop_recommend(  ):   
    if request.method == 'POST':
        data = request.json

        try :
         crop_name = get_crop(data)

        except Exception as e:
            print('errororo ' , e)
            return 'error' , 500

        return {'crop_name' :  crop_name}



@app.route("/api/fertilizer_recommend" , methods=['POST'])
def fertilizer_recommend( ):   
    if request.method == 'POST':
        features = request.json

        try:
            fertilizer_name = get_fertilizer(features)
        except :
            return jsonify({'error': str(e)}), 500

        return fertilizer_name
        

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        app.run(debug=True)