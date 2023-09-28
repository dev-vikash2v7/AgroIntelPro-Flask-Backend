import numpy as np
from tensorflow.keras.models import load_model
import json
from io import BytesIO
from PIL import Image

potato_disease_model = load_model('./saved_model/Disease_Prediction_Model/potato_disease_model.h5')
corn_disease_model = load_model('./saved_model/Disease_Prediction_Model/potato_disease_model.h5')

with open( "./saved_model/Disease_Prediction_Model/crop_disease_mapping.json", 'r') as json_file:
    crop_disease_mapping = json.load(json_file)


def get_pred_img(crop_type ,image_file ):

    if (crop_type == 'potato'):
        model = potato_disease_model 
    elif (crop_type == 'corn'):
            model = corn_disease_model 

    # Load and preprocess the uploaded image
    img = Image.open(BytesIO(image_file.read()))

    img = img.resize((64, 64))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    img_array = img_array / 225.0
    img_array = np.expand_dims(img_array, axis=0)

    # print(img_array.shape)#(1,64,64,3)

   # Make predictions using the model
    # predictions = model.predict(img_array)
    # # print('predictions  , ' , predictions)
    # predicted_class = np.argmax(predictions, axis=1)[0]
    # print('predicted_class  , ' , predicted_class)

    # print(crop_disease_mapping)

     # Return the predicted class label
    return crop_disease_mapping[crop_type][0]