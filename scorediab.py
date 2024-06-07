import json
import joblib
import numpy as np
from azureml.core.model import Model

# called the when the model is loaded
def init():
    global model
    #get the path to the deployed model file and load it
    model_path = Model.get_model_path('diabetes2_models')
    model = joblib.load(model_path)

#called when the request is recieved
def run(raw_data):
    #get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    #get a prediction from the model
    predictions = model.predict(data)
    #get the corresponding classnames for each prediction (0 or 1)
    classnames = ['not-diabetic', 'diabetic']
    predicted_classes =[]
    for prediction in predictions:
        predicted_classes.append(classnames[predictions])
        #return the predictions as Json
    return json.dumps(predicted_classes)
