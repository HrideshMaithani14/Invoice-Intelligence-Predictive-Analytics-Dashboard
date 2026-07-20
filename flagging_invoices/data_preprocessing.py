import pandas as pd
import sqlite3 as sql
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler

def load_invoice_data():
    cnn = sql.connect('D:\Programing\Programing_folder\Program Files\Real World projects of data science\Invoice Intelligence ML\data\inventory.db')
    query = """
        WITH  purchases_agg AS
        (select 
        p.PONumber,
        count(distinct p.Brand) as total_brand,
        sum(p.Quantity) as total_item_Quantity,
        sum(p.Dollars) as total_item_Dollars,
        avg(julianday(p.ReceivingDate)-julianday(p.PODate)) as avg_reciving_dely
        from purchases p
        group by p.PONumber)
                        
        select 
        v.PONumber,
        v.Quantity as invoice_Quantity,
        v.Dollars as invoice_Dollars,
        v.Freight,
        julianday(v.InvoiceDate)-julianday(v.PODate)   as day_po_to_invoice,             
        julianday(v.PayDate)-julianday(v.InvoiceDate)  as day_to_pay,
        pa.total_brand,
        pa.total_item_Quantity,
        pa.total_item_Dollars,
        pa.avg_reciving_dely

        from vendor_invoice as v
        LEFT JOIN purchases_agg as pa
            on v.PONumber = pa.PONumber
                        """
    df = pd.read_sql_query(query,cnn)
    cnn.close()
    return df
def create_invoice_flagging(row):
    if (abs(row['invoice_Dollars']-row['total_item_Dollars'])>5):
        return 1
    
    if (row['avg_reciving_dely']>10):
        return 1
    
    return 0

def apply_labels(df):
    df['flag_invoice']=df.apply(create_invoice_flagging,axis=1)
    return df

def splitting_data(df,feat,target):
    x=df[feat]
    y=df[target]
    return train_test_split(x,y,test_size=0.25,random_state=42,shuffle=True)

def scaled_features(X_train,X_test,Scaled_pth):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler,'Models\scaled_model\scaler.pkl')
    return X_train_scaled,X_test_scaled



    
