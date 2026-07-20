from catboost import CatBoostClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from xgboost import XGBClassifier 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.metrics import (accuracy_score, classification_report, 
                             precision_score, recall_score, f1_score, roc_auc_score)
import numpy as np

def eval_model(true, predicted, probas):
    accuracy_scor = accuracy_score(true, predicted)
    classification_repor = classification_report(true, predicted, zero_division=0)
    precision_scor = precision_score(true, predicted, average='weighted', zero_division=0)
    recall_scor = recall_score(true, predicted, average='weighted')
    f1_scor = f1_score(true, predicted, average='weighted')
    
    # Safely compute AUC handling potential binary vs multi-class shapes
    if len(np.unique(true)) > 2:
        roc_auc_scor = roc_auc_score(true, probas, multi_class='ovr', average='weighted')
    else:
        roc_auc_scor = roc_auc_score(true, probas[:, 1])
        
    return accuracy_scor, classification_repor, f1_scor, precision_scor, recall_scor, roc_auc_scor 

def model_training(X_train, X_test, y_train, y_test): 
    Xg_parem = { 
        'learning_rate': [0.01, 0.1, 0.2], 
        'max_depth': [5, 6, 8, 10], 
        'n_estimators': [100, 300, 500], 
        'colsample_bytree': [0.1, 0.5, 0.8, 1.0] 
    } 
    KNN_parem = { 
        'n_neighbors': [3, 5, 7, 9, 11, 15], 
        'weights': ['uniform', 'distance'], 
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
        'p': [1, 2] 
    } 
    Catboost_parem = { 
        'iterations': [100, 300, 500, 1000], 
        'learning_rate': [0.01, 0.1, 0.2], 
        'depth': [4, 6, 8, 10], 
        'l2_leaf_reg': [1, 3, 5, 7, 9] 
    } 
    
    models = [ 
        ('KNN', KNeighborsClassifier(), KNN_parem), 
        ('CatBoost', CatBoostClassifier(verbose=0), Catboost_parem), 
        ('XGBoost', XGBClassifier(), Xg_parem) 
    ] 
    
    best_model_name = None 
    classification_report_str = None 
    best_model_obj = None 
    best_accuracy = 0.0 
    
    print("Model Evaluation Begin\n" + "="*40)
    
    for model_name, model, parems in models: 
        # Exclude CatBoost from multi-processing to prevent deadlock
        current_n_jobs = None if model_name == 'CatBoost' else -1 
        
        random = RandomizedSearchCV(
            estimator=model, 
            param_distributions=parems, 
            n_iter=5, 
            verbose=1, 
            n_jobs=current_n_jobs, 
            cv=5, 
            random_state=42 
        ) 
        
        random.fit(X_train, y_train) 
        
        y_pred_test = random.predict(X_test) 
        y_pred_proba = random.predict_proba(X_test)
        
        testacc_sc, class_report_d, testf1_scor, testprec_scor, testrecall_sc, testroc_scoe = eval_model(
            y_test, y_pred_test, y_pred_proba
        ) 
        
        print(f"[{model_name}] Evaluation Results:") 
        print(f"Accuracy: {testacc_sc:.4f} | F1: {testf1_scor:.4f} | Precision: {testprec_scor:.4f} | AUC: {testroc_scoe:.4f}") 
        print(f"Classification Report:\n{class_report_d}")
        
        if testacc_sc > best_accuracy: 
            best_accuracy = testacc_sc 
            classification_report_str = class_report_d 
            best_model_name = model_name 
            best_model_obj = random.best_estimator_ 
            
        print("="*40)
        
    print("\n" + "="*40) 
    print(f"🏆 BEST MODEL: {best_model_name}") 
    print(f"🎯 Highest Accuracy: {best_accuracy:.4f}") 
    print(f'class_report:{classification_report_str}')
    print("="*40) 
    
    return random, best_model_name, best_model_obj, best_accuracy, classification_report_str




        
    
    
    
    