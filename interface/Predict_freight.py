import joblib
import pandas as pd
Model_pth = 'Models/best_models/freight.pkl'

def load_model(model_pth:str =Model_pth):
    with open(model_pth,'rb') as file:
        freight_model= joblib.load(file)
    return freight_model


def predict_freight_cost(input_data):
    model= load_model()
    input_df=pd.DataFrame(input_data)
    input_df['Predict_freight']=model.predict(input_df).round()
    return input_df
if __name__=='__main__':
    sample_data ={
        'Dollars':[18500,9000,3000,200]
    }
    prediction= predict_freight_cost(sample_data)
    print(prediction)