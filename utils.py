import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings("ignore")

import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, plot_importance as plot_importance_lgbm

import re
from scipy import stats
from tqdm import tqdm
import glob
import time

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV




def clean_text(text):
    "preprocess text data"
    
    text = str(text).lower()
    text = re.sub(r'\(.*', ' ', text)
    text = text.strip()
    text = re.sub(r'[ -]', '_', text)
    
    return text


def calculate_churn_risk_score(row):
    
    score = 0
    if row['Contract'] == 'month_to_month':
        score += 3
    if row['PaymentMethod'] == 'electronic_check':
        score += 2
    if row['PaperlessBilling'] == 'yes':
        score += 1
    if row['TechSupport'] < 'no':
        score += 2
    if row['DeviceProtection'] < 'no':
        score += 1
    if row['OnlineBackup'] < 'no':
        score += 1
    if row['OnlineSecurity'] < 'no':
        score += 2
    if row['InternetService'] < 'fiber_optic':
        score += 1
    if row['tenure'] < 12:
        score += 3
    return score


def map_target_data(data:pd.Series) -> pd.Series:
    """
    Preprocess and maps the target variable
    """
    
    data = data.apply(clean_text)

    with open(r'.\dumps\target_mapping.pkl', 'rb') as  f:
        target_mapping = pickle.load(f)

    return data.map(target_mapping)


def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    This function preprocess data and converts it into numerical format to be fed to model
    """

    # clean the categorical features if not cleaned
    with open(r'.\dumps\categorical_columns.pkl', 'rb') as f:
        categorical_cols = pickle.load(f)

    for feature in categorical_cols:
        df[feature] = df[feature].apply(clean_text)


    # Numeric variables
    with open(r'.\dumps\numeric_columns.pkl', 'rb') as f:
        numeric_cols = pickle.load(f)
    with open(r'.\dumps\numeric_FE_columns.pkl', 'rb') as f:
        numeric_FE_columns = pickle.load(f)
    

    # Feature Charges_to_tenure_ratio
    df['Charges_to_tenure_ratio'] = df['MonthlyCharges'] / df['tenure']
    with open(r'.\dumps\Charges_to_tenure_ratio_transformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    df['Charges_to_tenure_ratio_transformed'] = transformer.transform(df[['Charges_to_tenure_ratio']])

    # Feature tenure
    with open(r'.\dumps\tenure_transformer.pkl','rb') as f:
        transformer = pickle.load(f)
    df['tenure_transformed']= transformer.transform(df[['tenure']])
    
    # Feature MonthlyCharges
    with open(r'.\dumps\MonthlyCharges_transformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    df['MonthlyCharges_transformed']= transformer.transform(df[['MonthlyCharges']])

    # Feature TotalCharges
    with open(r'.\dumps\TotalCharges_transformer.pkl','rb') as f:
        transformer = pickle.load(f)
    df['TotalCharges_transformed'] = transformer.transform(df[['TotalCharges']])

    # Tenure Binning
    df.loc[(df["tenure"] <= 12), "New_tenure_Year"] = "0_1_Year"
    df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "New_tenure_Year"] = "1_2_Year"
    df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "New_tenure_Year"] = "2_3_Year"
    df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "New_tenure_Year"] = "3_4_Year"
    df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "New_tenure_Year"] = "4_5_Year"
    df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "New_tenure_Year"] = "5_6_Year"

    # People who do not receive any support, backup or protection
    df["New_Support"] = df.apply(lambda x: '1' if (x["OnlineBackup"] != "yes") or (x["DeviceProtection"] != "yes") or (x["TechSupport"] != "yes") else '0', axis=1)

    df['New_Churn_Risk_Score'] = df.apply(calculate_churn_risk_score, axis=1)
    
    # Categorical variables 
    with open(r'.\dumps\one_hot_features.pkl', 'rb') as f:
        one_hot_features = pickle.load(f)

    with open(r'.\dumps\mapping_features.pkl', 'rb') as f:
        mapping_features = pickle.load(f)


    # Mapping features
    with open(r'.\dumps\mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    for feature in mapping_features:
        df[feature] = df[feature].map(mapping).astype(int)

    # One-hot features
    with open(r'.\dumps\one_hot_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    ohe_features_encoded = pd.DataFrame(encoder.transform(df[one_hot_features]), columns= encoder.get_feature_names_out())
    ohe_features_encoded = ohe_features_encoded.astype(int)

    df_imb = pd.concat([df[numeric_cols + numeric_FE_columns + mapping_features], ohe_features_encoded], axis=1)
    
    return df_imb


def select_train_data(sample:str, scale:str, return_path=False):
    """
    allowed_samples = ['None', 'smote', 'adasyn']
    allowed_scales = ['None', 'standard', 'minmax']
    """
    
    allowed_samples = ['None', 'smote', 'adasyn']
    allowed_scales = ['None', 'standard', 'minmax']

    # Validate inputs
    if sample not in allowed_samples:
        raise ValueError(f"Invalid sample value. Allowed values are {allowed_samples}.")
    if scale not in allowed_scales:
        raise ValueError(f"Invalid scale value. Allowed values are {allowed_scales}.")

    sample_mapping = {'None':'imb', 'smote': 'smote', 'adasyn': 'adasyn'}
    scale_mapping = {'None': '', 'standard': 'std', 'minmax':'minmax'}

    sample_value = sample_mapping[sample]
    scale_value = scale_mapping[scale]

    if scale_value == '':
        path = glob.glob(f'.\\data_for_model\\train\\*{sample_value}_data.pkl')[0]
    else:
        path = glob.glob(f'.\\data_for_model\\train\\*{sample_value}*{scale_value}*.pkl')[0]
    
    with open(path, 'rb') as f:
        X_train, y_train = pickle.load(f)

    if return_path:
        return X_train, y_train, path
    else:
        return X_train, y_train


def load_all_feature_names():

    # Numeric
    with open(r'.\dumps\numeric_columns.pkl', 'rb') as f:
        numeric_cols = pickle.load(f)
    with open(r'.\dumps\numeric_FE_columns.pkl', 'rb') as f:
        numeric_FE_columns = pickle.load(f)
    
    num_cols = numeric_cols + numeric_FE_columns

    # categorical
    with open(r'.\dumps\mapping_features.pkl', 'rb') as f:
        mapping_features = pickle.load(f)
    with open(r'.\dumps\one_hot_features.pkl', 'rb') as f:
        one_hot_features = pickle.load(f)
    with open(r'.\dumps\one_hot_encoder_feature_names.pkl', 'rb') as f:
        one_hot_encoder_feature_names = pickle.load(f)

    return numeric_cols, numeric_FE_columns, num_cols, mapping_features, one_hot_features, one_hot_encoder_feature_names


def scale_test_data(sample:str, scale:str, data:list, return_path=False):

    """
    Scale test data using standard or minmax scaler used to fit smote sampled train data or adasyn sampled train data
    
    allowed_samples = ['None', 'smote', 'adasyn']
    allowed_scales = ['standard', 'minmax']
    data = list(X_test, y_test)
    """

    X_test = data[0]
    y_test = data[-1]

    allowed_samples = ['None', 'smote', 'adasyn']
    allowed_scales = ['standard', 'minmax']

    # Validate inputs
    if sample not in allowed_samples:
        raise ValueError(f"Invalid sample value. Allowed values are {allowed_samples}.")
    if scale not in allowed_scales:
        raise ValueError(f"Invalid scale value. Allowed values are {allowed_scales}.")

    sample_mapping = {'None': 'imb', 'smote': 'smote', 'adasyn': 'adasyn'}
    scale_mapping = {'standard': 'standardscaler', 'minmax':'minmaxscaler'}

    sample_value = sample_mapping[sample]
    scale_value = scale_mapping[scale]

    path = glob.glob(f'.\\dumps\\*{sample_value}*{scale_value}*.pkl')[0]
    

    # Scaling
    numeric_cols, numeric_FE_columns, num_cols, mapping_features, one_hot_features, one_hot_encoder_feature_names = load_all_feature_names()
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    
    num_scaled = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols)
    X_te_scaled = pd.concat([num_scaled, X_test[mapping_features], X_test[one_hot_encoder_feature_names]], axis=1)
    
    if return_path:
        return X_te_scaled, y_test, path
    else:
        return X_te_scaled, y_test


def load_models():

    # Model dictionary
    models = { 
        # Logistic Regression model
        "Logistic Regression": LogisticRegression(),
    
        # Naive Bayes model
        "Naive Bayes": GaussianNB(),

        # Support Vector Machine 
        "SVC": SVC(),
       
        # Decision Tree model
        "Decision Tree Classifier": DecisionTreeClassifier(),
    
        # Random Forest model
        "Random Forest": RandomForestClassifier(),
    
        # LightGBM model
        "LightGBM": LGBMClassifier()
    }

    return models


def load_model_params():
    
    model_params = {
        'Logistic Regression' : {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],
            'solver' : ['lbfgs', 'liblinear', 'newton-cg']},
    
        'Naive Bayes' : {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    
        'SVC' : {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
    
        'Decision Tree Classifier' : {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': np.arange(2, 20,2),
            'min_samples_leaf' : np.arange(2, 20,2),
            'max_features' : ['sqrt', 'log2']},
        
        'Random Forest' : {
            'criterion' : ['gini', 'entropy'],
            'n_estimators': [50, 100, 200, 500, 800, 1000, 1200, 1500, 1800],
            'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'min_samples_split': np.arange(2, 20, 2),
            'min_samples_leaf': np.arange(2, 20, 2),
            'bootstrap': [True, False],
            'max_samples' :[None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
    
        'LightGBM' : {
            'n_estimators': [50, 100, 200, 500, 800, 1000, 1200, 1500],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
            'num_leaves': [31, 50, 80, 100, 120, 150],
            'max_depth' : [-1, 2, 5, 10, 20, 30],
            'boosting_type': ['gbdt', 'dart', 'goss'],
            'min_child_samples' : np.arange(10, 121, 20),
            'feature_fraction' : np.arange(0.1,1,0.1)}
    }

    return model_params


def tune_model(method, model, param_grid, X_train, y_train, scoring='f1', n_iter=10, cv=5, random_state=42):

    """Tune a model using either GridSearchCV or RandomizedSearchCV and return the best model."""

    # Choose the tuning method based on the 'method' parameter
    if method == 'grid':
        search = GridSearchCV(model, param_grid, scoring=scoring, refit='accuracy', cv=cv, n_jobs=-1, verbose=1)
    elif method == 'random':
        search = RandomizedSearchCV(model, param_distributions=param_grid, scoring=scoring, refit='accuracy', n_iter=n_iter, cv=cv, n_jobs=-1, verbose=1, random_state=random_state)
    else:
        raise ValueError("Invalid method specified. Use 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.")

    # Fit the search to the training data
    search.fit(X_train, y_train)

    # Return the best model
    return search.best_estimator_, search.best_score_


def train_models(train_data:str, test_data:str, train_feature_sets, test_feature_sets, method='random', tune=False, scoring='f1', models:dict=None) -> dict:

    """
    models to be passed as dictionary with name as key and model as value
    models = { 
        "Logistic Regression": LogisticRegression(), 
        "Naive Bayes": GaussianNB()
    """
    
    X_train, y_train = train_feature_sets[train_data]
    X_test, y_test = test_feature_sets[test_data]

    if models is None:
        models = load_models()
    
    results = []
    for name, model in tqdm(models.items()):

        if tune:
            model_params = load_model_params()
            model, score = tune_model(method, model, param_grid = model_params[name], X_train=X_train, y_train=y_train, 
                           scoring=scoring, n_iter=10, cv=5, random_state=42)
        
        # Model training
        model.fit(X_train, y_train)
        y_tr_pred = model.predict(X_train)
        y_te_pred = model.predict(X_test)
        
        # Model Evaluation
        train_f1_micro = f1_score(y_train, y_tr_pred, average='micro')
        test_f1_micro = f1_score(y_test, y_te_pred, average='micro')
        train_f1_macro = f1_score(y_train, y_tr_pred, average='macro')
        test_f1_macro = f1_score(y_test, y_te_pred, average='macro')
        train_f1_weighted = f1_score(y_train, y_tr_pred, average='weighted')
        test_f1_weighted = f1_score(y_test, y_te_pred, average='weighted')
        
        # append the scores
        results.append(
            {'Model': name, 
            'train_data': train_data, 'test_data': test_data,
            'train_f1_micro':round(train_f1_micro,4), 'test_f1_micro':round(test_f1_micro,4),
            'train_f1_macro':round(train_f1_macro,4), 'test_f1_macro':round(test_f1_macro,4), 
            'train_f1_weighted':round(train_f1_weighted,4), 'test_f1_weighted':round(test_f1_weighted,4),
            }
        )

    return results


def evaluate_model(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)
    y_te_pred = model.predict(X_test)
        
    # Model Evaluation

    # f1-score
    train_f1_micro = f1_score(y_train, y_tr_pred, average='micro')
    test_f1_micro = f1_score(y_test, y_te_pred, average='micro')
    
    print('micro f1_score- train_f1_micro:', round(train_f1_micro,4), 'test_f1_micro:', round(test_f1_micro,4), '\n')

    # Classification report
    report = classification_report(y_train, y_tr_pred)
    print('Classification Report on train data:\n', report, '\n')

    report = classification_report(y_test, y_te_pred)
    print('Classification Report on test data:\n', report)
    
    # confusion matrix
    cm_tr = confusion_matrix(y_train, y_tr_pred)
    cm_tr_df = pd.DataFrame(cm_tr) #, index=, columns=iris.target_names)
    
    cm_te = confusion_matrix(y_test, y_te_pred)
    cm_te_df = pd.DataFrame(cm_te) #, index=, columns=iris.target_names)

    fig, axs = plt.subplots(1,2, figsize=(9,3))
    sns.heatmap(cm_tr_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axs[0])
    axs[0].set_title('Train Confusion Matrix')
    axs[0].set_xlabel('Predicted Label')
    axs[0].set_ylabel('True Label')
        
    sns.heatmap(cm_te_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axs[1])
    axs[1].set_title('Test Confusion Matrix')
    axs[1].set_xlabel('Predicted Label')
    axs[1].set_ylabel('True Label')
    print('\n')

    ## roc_auc curve
    y_tr_proba = model.predict_proba(X_train)[:,1]
    y_te_proba = model.predict_proba(X_test)[:,1]

    # Calculate ROC curve and ROC AUC for train data
    fpr_train, tpr_train, _ = roc_curve(y_train, y_tr_proba)
    roc_auc_train = roc_auc_score(y_train, y_tr_proba)
    
    # Calculate ROC curve and ROC AUC for test data
    fpr_test, tpr_test, _ = roc_curve(y_test, y_te_proba)
    roc_auc_test = roc_auc_score(y_test, y_te_proba)
    
    # Plot ROC Curves
    plt.figure(figsize=(6, 4))
    plt.plot(fpr_train, tpr_train, label=f'Train ROC AUC = {roc_auc_train:.2f}')
    plt.plot(fpr_test, tpr_test, label=f'Test ROC AUC = {roc_auc_test:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    
    ## precision_recall curve
    y_tr_proba = model.predict_proba(X_train)[:,1]
    y_te_proba = model.predict_proba(X_test)[:,1]

    # Calculate Precision-Recall curve for train data
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_tr_proba)
    auc_pr_train = auc(recall_train, precision_train)
    
    # Calculate Precision-Recall curve for test data
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_te_proba)
    auc_pr_test = auc(recall_test, precision_test)
    
    # Plot Precision-Recall Curves
    plt.figure(figsize=(6, 4))
    plt.plot(recall_train, precision_train, label=f'Train AUC = {auc_pr_train:.2f}')
    plt.plot(recall_test, precision_test, label=f'Test AUC = {auc_pr_test:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


