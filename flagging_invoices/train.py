from data_preprocessing import *
from Model_tranning import *
import joblib

features = [
    "invoice_Quantity",
    "invoice_Dollars",
    "Freight",
    "total_item_Quantity",
    "total_item_Dollars"
]

target_feat = 'flag_invoice'

def main():
    print("🚀 Starting the classification pipeline...")
    
    #load_data
    print("⏳ Loading invoice data and applying labels...")
    df = load_invoice_data()
    df = apply_labels(df)
    
    ## trainning test slpit data
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = splitting_data(df, features, target_feat)

    print("📏 Scaling features...")
    Scaled_pth='Models\scaled_model\scaler.pkl'
    X_train_scaled, X_test_scaled = scaled_features(X_train, X_test,Scaled_pth)
    
    #traniing predicting data 
    print("🧠 Traniing models and evaluating...")
    random, best_name, best_model, best_acc ,class_report= model_training(X_train_scaled, X_test_scaled, y_train, y_test)
    
    print(f"🏆 Best Model Found: {best_name} with Accuracy: {best_acc}")
    
    #save best model
    print("💾 Extracting best estimator and saving to disk...")
    best_model = random.best_estimator_
    path = 'Models/best_models\Flagged_invoice.pkl'
    joblib.dump(best_model, path)
    print(f"✅ Success! Model saved to '{path}'")

if __name__ == "__main__":
    main()