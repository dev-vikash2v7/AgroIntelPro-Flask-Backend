import os
from fastapi import FastAPI , File , UploadFile 
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.ndimage import zoom

current_directory = os.path.dirname(os.path.abspath(__file__))

# import gdown

# def download_model():
    # # Replace 'your_file_id' with the actual file ID
    # file_id = 'your_file_id'

    # # Define the output file name
    # output_file = 'downloaded_file.ext'  # Replace with the desired file name and extension

    # # Construct the download URL
    # url = f'https://drive.google.com/uc?id={file_id}'

    # # Download the file
    # gdown.download(url, output_file, quiet=False)



PotatoDiseaseModel = load_model(current_directory + '/saved_model/potato_disease_model.h5')
with open(current_directory + "/saved_model/potato_disease_mapping.json", 'r') as json_file:
    class_names = json.load(json_file)

app = FastAPI()


def get_pred_img(model ,img ):

    # img = image.load_img(img_path, target_size=(64, 64))
    img_bytes = Image.open(BytesIO(img)) 

    img_array = np.array(img_bytes)
    img_array =  np.resize( img_array , (64 , 64 , 3))
    img_array = img_array / 225.0

    img_array = np.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)

    pred_array = model.predict(img_array)[0]
    print(pred_array)

    result = {}

    for label, confidence in zip(class_names , pred_array):
            result[label] = np.round(100*confidence , 2)

    return result


@app.get("/")
async def root():
    return {"message": "Hello World from potato disease model"}


@app.post("/predict/{crop}")
async def predict(crop ,  file : UploadFile = File(...)   ):   
    
    img  = await file.read()

    if (crop == 'potato'):
        model = PotatoDiseaseModel 

    result = get_pred_img( model , img )

    return result
