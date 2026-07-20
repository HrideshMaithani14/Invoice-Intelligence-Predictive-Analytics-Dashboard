import joblib
import pandas as pd
import os

# Ensure this path is correct relative to where you run app.py
Model_Pth = 'Models/best_models/Flagged_invoice.pkl'

def load_model(model_pth: str = Model_Pth):
    with open(model_pth, 'rb') as file:
        invoice_flag_model = joblib.load(file)
    return invoice_flag_model

def predict_flag_invoice(input_data):
    model = load_model()
    
    # REMOVED the .T here. 
    # input_data is a dict of lists, so pd.DataFrame natively maps it to 1 row with 5 columns.
    input_df = pd.DataFrame(input_data)
    
    # Generate prediction and append it as a new column
    input_df['Predict_flag'] = model.predict(input_df).round()
    
    return input_df