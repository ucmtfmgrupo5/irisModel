import pickle
import os
import dill
import pandas as pd
import json

class Prediction:

    model = None

    def __init__(self):
        currrentPath = os.path.dirname(os.path.abspath(__file__))
        parentPath = os.path.abspath(os.path.join(currrentPath, os.pardir))
        filename = parentPath+'/modelSampleModel/modelSampleModel'
        print(filename)
        infile = open(filename,'rb')
        self.model = pickle.load(infile)
        infile.close()


    def makePrediction(self, inputDF):

        #Transform input DF into proper format
        inputModel = inputDF.values

        #Predict values from input
        predictions = self.model.predict(inputModel)


        #Transform output prediction into dataframe
        predictionDF = pd.DataFrame(predictions, columns=['prediction'])

        #Build output response object 
        Body = []
        for index, row in inputDF.iterrows():
            sepal_length = row['sepal-length']
            sepal_width = row['sepal-width']
            petal_length = row['petal-length']
            petal_width = row['petal-width']

            prediction = predictionDF.iloc[index]['prediction']

            InputVals = {}

            InputVals['sepal-length'] = sepal_length
            InputVals['sepal-width'] = sepal_width
            InputVals['petal-length'] = petal_length
            InputVals['petal-width'] = petal_width

            PredictionBody = {}

            PredictionBody['input'] = InputVals
            PredictionBody['prediction'] = prediction

            Body.append(PredictionBody)

            output = json.dumps(Body)

        
        return output


if __name__ == "__main__":
    predictionObject = Prediction()
    currrentPath = os.path.dirname(os.path.abspath(__file__))   
    parentPath = os.path.abspath(os.path.join(currrentPath, os.pardir))
    filenameDill = parentPath+'/predictionSampleModel/prediction'
    with open(filenameDill, "wb") as f:
     dill.dump(predictionObject, f)
