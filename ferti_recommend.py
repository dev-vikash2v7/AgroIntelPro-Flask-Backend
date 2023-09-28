import joblib 


ferti_encode = joblib.load('./saved_model/Ferti_Recommend_Model/encode_ferti.pkl')
crop_encode = joblib.load('./saved_model/Ferti_Recommend_Model/encode_crop.pkl')
soil_encode = joblib.load('./saved_model/Ferti_Recommend_Model/encode_soil.pkl')
model = joblib.load('./saved_model/Ferti_Recommend_Model/RF_Model.pkl')


def get_fertilizer( data  ):   

        # data['NPK Total'] = data['N'] + data['P'] + data['K']
        # data['Nitrogen (%)'] = (data['N'] / data['NPK Total']) * 100
        # data['Phosphorus (%)'] = (data['P'] / data['NPK Total']) * 100
        # data['Potassium (%)'] = (data['K'] / data['NPK Total']) * 100

        feature = {
        "Temparature"     :   data['temp'],
        "Humidity"         : data['humidity'],
        "Moisture"             : data['moisture'],
        "Soil Type"          : data['soil_type'],
        "Crop Type"         : data['crop_type'],
        "Nitrogen"           :   data['N'],
        "Potassium"         :     data['K'],
        "Phosphorous"         :   data['P']
        }



        feature['Soil Type'] = soil_encode.transform([feature['Soil Type']])[0]
        feature['Crop Type'] = crop_encode.transform([feature['Crop Type']])[0]


        features_arr = list(feature_new.values())

        predictions_encoded = model.predict([features_arr])
        ferti_name = ferti_encode.inverse_transform(predictions_encoded)[0]

        print(ferti_name)

        return ferti_name

