# -*- coding: utf-8 -*-
print("------------------------------------------------------")
print("---------------- Metadata Information ----------------")
print("------------------------------------------------------")
print("")

print("In the name of God")
print("Project: AutoCML: Automatic Comparative Machine Learning")
print("Creator: Mohammad Reza Saraei")
print("Contact: mrsaraei@yahoo.com")
print("Supervisor: Dr. Saman Rajebi")
print("Created Date: May 20, 2022")
print("") 

print("----------------------------------------------------")
print("------------------ Import Libraries ----------------")
print("----------------------------------------------------")
print("")

# Import Libraries for Python
import pandas as pd
import numpy as np
import random
import os
from pandas import set_option
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, f1_score, recall_score
import warnings
warnings.filterwarnings("ignore")

print("----------------------------------------------------")
print("------------------ Data Ingestion ------------------")
print("----------------------------------------------------")
print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv('RawData.csv')

# print("----------------------------------------------------")
# print("----------------- Set Option -----------------------")
# print("----------------------------------------------------")
# print("")

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 1000)

# print("------------------------------------------------------")
# print("-------------- Tune-up Seed for ML Models ------------")
# print("------------------------------------------------------")
# print("")

# Set a Random State value
RANDOM_STATE = 42

# Set Python random a fixed value
random.seed(RANDOM_STATE)

# Set environment a fixed value
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

# Set numpy random a fixed value
np.random.seed(RANDOM_STATE)

print("------------------------------------------------------")
print("------------ Initial Data Understanding --------------")
print("------------------------------------------------------")
print("")

print("Initial General Information:")
print("****************************")
print(df.info())
print("")

print("------------------------------------------------------")
print("---------------- Data Label Encoding -----------------")
print("------------------------------------------------------")
print("")

# Encoding Coulmns Having Objects by LabelEncoder
obj = df.select_dtypes(include = ['object'])
LE = preprocessing.LabelEncoder()
col = obj.apply(LE.fit_transform)

print("Columns Having Objects:")
print("***********************")
print(obj.head(10))
print("")

print("Encoding Columns Having Object:")
print("*******************************")
print(col.head(10))
print("")
print('Shape of Encoded Columns:', col.shape)
print("")

print("------------------------------------------------------")
print("------------- Save Encoded Objects Data --------------")
print("------------------------------------------------------")
print("")

# Save DataFrame After Encoding
pd.DataFrame(col).to_csv('EncodedData.csv', index = False)

print("------------------------------------------------------")
print("------- Creating Main DataFrame by Combination -------")
print("------------------------------------------------------")
print("")

# Import Encoded Objects DataFrame (.csv) by Pandas Library
df_col = pd.read_csv('EncodedData.csv')

# Combinating Encoded Data with Main DataFrame
df_obj = df.drop(df.select_dtypes(include = ['object']), axis = 1)

print("Columns' Name that needs to encoding:", obj.columns)
print("")

print("The Target Column Name:", df.columns[-1])
print("")

if df.columns[-1] in obj.columns:
    df = pd.concat([df_obj, df_col], axis = 1)
else:
    df = pd.concat([df_col, df_obj], axis = 1)

print("An overview of Encoded Data:")
print("****************************")
print("")
print(df.head(1))
print("")

print("------------------------------------------------------")
print("--------- Data Understanding After Encoding ----------")
print("------------------------------------------------------")
print("")

print("General Information After Encoding:")
print("***********************************")
print(df.info())
print("")

print("------------------------------------------------------")
print("------------------ Data Spiliting --------------------")
print("------------------------------------------------------")
print("")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values     

print("------------------------------------------------------")
print("---------------- Data Normalization ------------------")
print("------------------------------------------------------")
print("")

# Normalization [0, 1] of Data
scaler = MinMaxScaler(feature_range = (0, 1))
f = scaler.fit_transform(f)
print(f)
print("")

print("------------------------------------------------------")
print("----------- Save Features and Target Data ------------")
print("------------------------------------------------------")
print("")

# Save DataFrame (f, t) After Munging
pd.DataFrame(f).to_csv('FeaturesData.csv', index = False)
pd.DataFrame(t).to_csv('TargetData.csv', index = False)

print("------------------------------------------------------")
print("-------- Features and Target Data Combination --------")
print("------------------------------------------------------")
print("")

# Import Again DataFrames (f, t) by Pandas Library
df_f = pd.read_csv('FeaturesData.csv')
df_t = pd.read_csv('TargetData.csv')

# Rename t Column
df_t.rename(columns = {'0': 'Diagnosis'}, inplace = True)

# Combination of DataFrames
df = pd.concat([df_f, df_t], axis = 1)

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv('MainDataFrame.csv', index = False)

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv('MainDataFrame.csv')

print("------------------------------------------------------")
print("---------------- Data Preprocessing ------------------")
print("------------------------------------------------------")
print("")

# Replace Question Mark to NaN:
df.replace("?", np.nan, inplace = True)

# Remove Duplicate Samples
df = df.drop_duplicates()
print("Duplicate Records After Removal:", df.duplicated().sum())
print("")

# Replace Mean instead of Missing Values
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp.fit(df)
df = imp.transform(df)
print("Mean Value For NaN Value:", "{:.3f}".format(df.mean()))
print("")

# Reordering Records / Samples / Rows
print("Reordering Records:")
print("*******************")
df = pd.DataFrame(df).reset_index(drop = True)
print(df)
print("")

print("------------------------------------------------------")
print("------------------ Data Respiliting ------------------")
print("------------------------------------------------------")
print("")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values     

print("------------------------------------------------------")
print("----------------- Outliers Detection -----------------")
print("------------------------------------------------------")
print("")

# Identify Outliers in the Training Data
ISF = IsolationForest(n_estimators = 100, contamination = 0.1, bootstrap = True, n_jobs = -1)

# Fitting Outliers Algorithms on the Training Data
ISF = ISF.fit_predict(f, t)

# Select All Samples that are not Outliers
Mask = ISF != -1
f, t = f[Mask, :], t[Mask]

print('nFeature:', f.shape)
print('nTarget:', t.shape)
print("")

print("------------------------------------------------------")
print("------------- Data Balancing By SMOTE ----------------")
print("------------------------------------------------------")
print("")

# Summarize Targets Distribution
print('Targets Distribution Before SMOTE:', sorted(Counter(t).items()))

# OverSampling (OS) Fit and Transform the DataFrame
OS = SMOTE()
f, t = OS.fit_resample(f, t)

# Summarize the New Targets Distribution
print('Targets Distribution After SMOTE:', sorted(Counter(t).items()))
print("")

print('nFeature:', f.shape)
print('nTarget:', t.shape)
print("")

print("------------------------------------------------------")
print("----------- Save Features and Target Data ------------")
print("------------------------------------------------------")
print("")

# Save DataFrame (f, t) After Munging
pd.DataFrame(f).to_csv('FeaturesData.csv', index = False)
pd.DataFrame(t).to_csv('TargetData.csv', index = False)

print("------------------------------------------------------")
print("-------- Features and Target Data Combination --------")
print("------------------------------------------------------")
print("")

# Import Again DataFrames (f, t) by Pandas Library
df_f = pd.read_csv('FeaturesData.csv')
df_t = pd.read_csv('TargetData.csv')

# Rename t Column
df_t.rename(columns = {'0': 'Diagnosis'}, inplace = True)

# Combination of DataFrames
df = pd.concat([df_f, df_t], axis = 1)

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv('MainDataFrame.csv', index = False)

print("------------------------------------------------------")
print("----------------- Data Understanding -----------------")
print("------------------------------------------------------")
print("")

print("Dataset Overview:")
print("*****************")
print(df.head(10))
print("")

print("General Information:")
print("********************")
print(df.info())
print("")

print("Statistics Information:")
print("***********************")
print(df.describe(include="all"))
print("")

print("nSample & (nFeature + Target):", df.shape)
print("")

print("Samples Range:", df.index)
print("")

print(df.columns)
print("")

print("Missing Values (NaN):")
print("*********************")
print(df.isnull().sum())                                         
print("")

print("Duplicate Records:", df.duplicated().sum())
print("")   

print("Features Correlations:")
print("**********************")
print(df.corr(method='pearson'))
print("")

print("------------------------------------------------------")
print("--------------- Data Distribution --------------------")
print("------------------------------------------------------")
print("")

print("nSample & (nFeature + Target):", df.shape)
print("")

print("Skewed Distribution of Features:")
print("********************************")
print(df.skew())
print("")
print(df.dtypes)
print("")

print("Target Distribution:")
print("********************")
print(df.groupby(df.iloc[:, -1].values).size())
print("")

print("------------------------------------------------------")
print("----------- Plotting Distribution of Data ------------")
print("------------------------------------------------------")
print("")

# Plot the Scores by Descending
plt.hist(df)
plt.xlabel('Data Value', fontsize = 11)
plt.ylabel('Data Frequency', fontsize = 11)
plt.title('Data Distribution After Preparation')
plt.savefig('AutoDP_DataDistribution.png', dpi = 600)
plt.savefig('AutoDP_DataDistribution.tif', dpi = 600)
plt.show()
plt.close()

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv('MainDataFrame.csv')

print("------------------------------------------------------")
print("------------------ Data Respiliting ------------------")
print("------------------------------------------------------")
print("")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values

print("------------------------------------------------------")
print("------------------ Test & Train Data -----------------")
print("------------------------------------------------------")
print("")

# Split Train and Test Data in Proportion of 70:30 %
f_train, f_test, t_train, t_test = train_test_split(f, t, test_size = 0.33, random_state = RANDOM_STATE)

print('Feature Train Set:', f_train.shape)
print('Feature Test Set:', f_test.shape)
print('Target Train Set:', t_train.shape)
print('Target Test Set:', t_test.shape)
print("")

print("------------------------------------------------------")
print("----------------- ML Models Building -----------------")
print("------------------------------------------------------")
print("")

print("KNN = KNeighborsClassifier")
print("DTC = DecisionTreeClassifier")
print("GNB = GaussianNBClassifier")
print("SVM = SupportVectorMachineClassifier")
print("LRG = LogisticRegressionClassifier")
print("MLP = MLPClassifier")
print("RFC = RandomForestClassifier")
print("GBC = GradientBoostingClassifier")
print("XGB = XGBClassifier")
print("ADB = AdaBoostClassifier")
print("ETC = ExtraTreesClassifier")
print("CBC = CatBoostClassifier")
print("")

# Creating Machine Learning Models
KNN = KNeighborsClassifier(n_neighbors = 6, p = 2)
DTC = DecisionTreeClassifier(random_state = RANDOM_STATE)
GNB = GaussianNB()
SVM = SVC(decision_function_shape = "ovo", probability = True, random_state = RANDOM_STATE)
LRG = LogisticRegression(solver ='lbfgs', random_state = RANDOM_STATE)
MLP = MLPClassifier(max_iter = 500, solver = 'lbfgs', random_state = RANDOM_STATE)
RFC = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = RANDOM_STATE)
GBC = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.5, random_state = RANDOM_STATE)
XGB = XGBClassifier(n_estimators = 100, eval_metric = 'error', objective = 'binary:logistic')
ADB = AdaBoostClassifier(n_estimators = 100, random_state = RANDOM_STATE)
ETC = ExtraTreesClassifier(n_estimators = 100, random_state = RANDOM_STATE)
CBC = CatBoostClassifier(iterations = 10, learning_rate = 0.5, loss_function = 'MultiClass')

# Fitting Machine Learning Models on Train Data 
KNN.fit(f_train, t_train)
DTC.fit(f_train, t_train)
GNB.fit(f_train, t_train)
SVM.fit(f_train, t_train)
LRG.fit(f_train, t_train)
MLP.fit(f_train, t_train)
RFC.fit(f_train, t_train)
GBC.fit(f_train, t_train)
XGB.fit(f_train, t_train)
ADB.fit(f_train, t_train)
ETC.fit(f_train, t_train)
CBC.fit(f_train, t_train)

# Prediction of Test Data by Machine Learning Models 
t_pred0 = KNN.predict(f_test)
t_pred1 = DTC.predict(f_test)
t_pred2 = GNB.predict(f_test)
t_pred3 = SVM.predict(f_test)
t_pred4 = LRG.predict(f_test)
t_pred5 = MLP.predict(f_test)
t_pred6 = RFC.predict(f_test)
t_pred7 = GBC.predict(f_test)
t_pred8 = XGB.predict(f_test)
t_pred9 = ADB.predict(f_test)
t_pred10 = ETC.predict(f_test)
t_pred11 = CBC.predict(f_test)

# Prediction of Test Data by Machine Learning Models for ROC_AUC_Score
t_pred0_prob = KNN.predict_proba(f_test)
t_pred1_prob = DTC.predict_proba(f_test)
t_pred2_prob = GNB.predict_proba(f_test)
t_pred3_prob = SVM.predict_proba(f_test)
t_pred4_prob = LRG.predict_proba(f_test)
t_pred5_prob = MLP.predict_proba(f_test)
t_pred6_prob = RFC.predict_proba(f_test)
t_pred7_prob = GBC.predict_proba(f_test)
t_pred8_prob = XGB.predict_proba(f_test)
t_pred9_prob = ADB.predict_proba(f_test)
t_pred10_prob = ETC.predict_proba(f_test)
t_pred11_prob = CBC.predict_proba(f_test)

print("")
print("------------------------------------------------------")
print("----------------- Accessed Results -------------------")
print("------------------------------------------------------")
print("")

# Machine Learning Models Overfitting-Underfitting Values
print("KNN Overfitting-Underfitting Value:", "{:.3f}".format(((KNN.score(f_train, t_train))-(KNN.score(f_test, t_test)))))
print("DTC Overfitting-Underfitting Value:", "{:.3f}".format(((DTC.score(f_train, t_train))-(DTC.score(f_test, t_test)))))
print("GNB Overfitting-Underfitting Value:", "{:.3f}".format(((GNB.score(f_train, t_train))-(GNB.score(f_test, t_test)))))
print("SVM Overfitting-Underfitting Value:", "{:.3f}".format(((SVM.score(f_train, t_train))-(SVM.score(f_test, t_test)))))
print("LRG Overfitting-Underfitting Value:", "{:.3f}".format(((LRG.score(f_train, t_train))-(LRG.score(f_test, t_test)))))
print("KNN Overfitting-Underfitting Value:", "{:.3f}".format(((MLP.score(f_train, t_train)) - (MLP.score(f_test, t_test)))))
print("RFC Overfitting-Underfitting Value:", "{:.3f}".format(((RFC.score(f_train, t_train)) - (RFC.score(f_test, t_test)))))
print("GBC Overfitting-Underfitting Value:", "{:.3f}".format(((GBC.score(f_train, t_train)) - (GBC.score(f_test, t_test)))))
print("XGB Overfitting-Underfitting Value:", "{:.3f}".format(((XGB.score(f_train, t_train)) - (XGB.score(f_test, t_test)))))
print("ADB Overfitting-Underfitting Value:", "{:.3f}".format(((ADB.score(f_train, t_train)) - (ADB.score(f_test, t_test)))))
print("ETC Overfitting-Underfitting Value:", "{:.3f}".format(((ETC.score(f_train, t_train)) - (ETC.score(f_test, t_test)))))
print("CBC Overfitting-Underfitting Value:", "{:.3f}".format(((CBC.score(f_train, t_train)) - (CBC.score(f_test, t_test)))))
print("")

print("------------------------------------------------------")
print("----------------- Confusion Matrix -------------------")
print("------------------------------------------------------")
print("")

# Calculating Confusion Matrix for Machine Learning Models
print("KNN CM:")
print(confusion_matrix(t_test, t_pred0))
print("DTC CM:")
print(confusion_matrix(t_test, t_pred1))
print("GNB CM:")
print(confusion_matrix(t_test, t_pred2))
print("SVM CM:")
print(confusion_matrix(t_test, t_pred3))
print("LRG CM:")
print(confusion_matrix(t_test, t_pred4))
print("MLP CM:")
print(confusion_matrix(t_test, t_pred5))
print("RFC CM:")
print(confusion_matrix(t_test, t_pred6))
print("GBC CM:")
print(confusion_matrix(t_test, t_pred7))
print("XGB CM:")
print(confusion_matrix(t_test, t_pred8))
print("ADB CM:")
print(confusion_matrix(t_test, t_pred9))
print("ETC CM:")
print(confusion_matrix(t_test, t_pred10))
print("CBC CM:")
print(confusion_matrix(t_test, t_pred11))

print("")

print("------------------------------------------------------")
print("----------------- Assessment Report ------------------")
print("------------------------------------------------------")
print("")

# Create DataFrames of Machine Learning Models
Models = ['KNN', 'DTC', 'GNB', 'SVM', 'LRG', 'MLP', 'RFC', 'GBC', 'XGB', 'ADB', 'ETC', 'CBC']

Accuracy = [accuracy_score(t_test, t_pred0),
            accuracy_score(t_test, t_pred1),
            accuracy_score(t_test, t_pred2),
            accuracy_score(t_test, t_pred3),
            accuracy_score(t_test, t_pred4),
            accuracy_score(t_test, t_pred5),
            accuracy_score(t_test, t_pred6),
            accuracy_score(t_test, t_pred7),
            accuracy_score(t_test, t_pred8),
            accuracy_score(t_test, t_pred9),
            accuracy_score(t_test, t_pred10),
            accuracy_score(t_test, t_pred11)]

ROC_AUC = [roc_auc_score(t_test, t_pred0_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred1_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred2_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred3_prob, multi_class = 'ovo'),               
           roc_auc_score(t_test, t_pred4_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred5_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred6_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred7_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred8_prob, multi_class = 'ovo'),               
           roc_auc_score(t_test, t_pred9_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred10_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred11_prob, multi_class = 'ovo')]

Precision = [precision_score(t_test, t_pred0, average = 'macro'),
             precision_score(t_test, t_pred1, average = 'macro'),
             precision_score(t_test, t_pred2, average = 'macro'),
             precision_score(t_test, t_pred3, average = 'macro'),
             precision_score(t_test, t_pred4, average = 'macro'),
             precision_score(t_test, t_pred5, average = 'macro'),
             precision_score(t_test, t_pred6, average = 'macro'),
             precision_score(t_test, t_pred7, average = 'macro'),
             precision_score(t_test, t_pred8, average = 'macro'),
             precision_score(t_test, t_pred9, average = 'macro'),
             precision_score(t_test, t_pred10, average = 'macro'),
             precision_score(t_test, t_pred11, average = 'macro')]

F1_Score = [f1_score(t_test, t_pred0, average = 'macro'),
            f1_score(t_test, t_pred1, average = 'macro'),
            f1_score(t_test, t_pred2, average = 'macro'),
            f1_score(t_test, t_pred3, average = 'macro'),
            f1_score(t_test, t_pred4, average = 'macro'),
            f1_score(t_test, t_pred5, average = 'macro'),
            f1_score(t_test, t_pred6, average = 'macro'),
            f1_score(t_test, t_pred7, average = 'macro'),
            f1_score(t_test, t_pred8, average = 'macro'),
            f1_score(t_test, t_pred9, average = 'macro'),
            f1_score(t_test, t_pred10, average = 'macro'),
            f1_score(t_test, t_pred11, average = 'macro')]

Recall = [recall_score(t_test, t_pred0, average = 'macro'),
          recall_score(t_test, t_pred1, average = 'macro'),
          recall_score(t_test, t_pred2, average = 'macro'),
          recall_score(t_test, t_pred3, average = 'macro'),
          recall_score(t_test, t_pred4, average = 'macro'),
          recall_score(t_test, t_pred5, average = 'macro'),
          recall_score(t_test, t_pred6, average = 'macro'),
          recall_score(t_test, t_pred7, average = 'macro'),
          recall_score(t_test, t_pred8, average = 'macro'),
          recall_score(t_test, t_pred9, average = 'macro'),
          recall_score(t_test, t_pred10, average = 'macro'),
          recall_score(t_test, t_pred11, average = 'macro')]

KFCV = [np.mean(cross_val_score(KNN, f, t, cv = 10)),
        np.mean(cross_val_score(DTC, f, t, cv = 10)),
        np.mean(cross_val_score(GNB, f, t, cv = 10)),
        np.mean(cross_val_score(SVM, f, t, cv = 10)),
        np.mean(cross_val_score(LRG, f, t, cv = 10)),
        np.mean(cross_val_score(MLP, f, t, cv = 10)),
        np.mean(cross_val_score(RFC, f, t, cv = 10)),
        np.mean(cross_val_score(GBC, f, t, cv = 10)),
        np.mean(cross_val_score(XGB, f, t, cv = 10)),
        np.mean(cross_val_score(ADB, f, t, cv = 10)),
        np.mean(cross_val_score(ETC, f, t, cv = 10)),
        np.mean(cross_val_score(CBC, f, t, cv = 10))]

model_scores = {'Model': Models,
                'Accuracy': Accuracy,
                'ROC-AUC': ROC_AUC,
                'Precision': Precision,
                'F1-Score': F1_Score,
                'Recall': Recall,
                'KFCV': KFCV}

print("")

df_models = pd.DataFrame(model_scores)
print(df_models)
print("")

# Adding 'Mean Scores' Column to Machine Learning Model Scores DataFrame
nModel = len(Models)
df_Mean = pd.DataFrame(df_models.iloc[:, 1: -1].mean(axis = 1))
df_models['Mean Scores'] = df_Mean

# Prioritized Machine Learning Models
prioritized_models = df_models.nlargest(nModel, 'Mean Scores')
print(prioritized_models)
print("")

print("----------------------------------------------------")
print("---------------- Plotting Outputs ------------------")
print("----------------------------------------------------")
print("")

# Plot the Scores by Descending
plt.barh(prioritized_models['Model'], prioritized_models['Mean Scores'], color = 'black')
plt.xlabel('Model', fontsize = 12)
plt.ylabel('Score', fontsize = 12)
plt.title('Prioritized Machine Learning Models based on AutoCML')
plt.savefig('AutoCML.png', dpi = 600)
plt.savefig('AutoCML.tif', dpi = 600)
plt.show()
plt.close()

print("-----------------------------------------------------")
print("---------------- The Best AutoCML Model -------------")
print("-----------------------------------------------------")
print("")

# Selection of the Best Machine Learning Model by Hybrid Descended Mean Scores
print(df_models.nlargest(1, 'Mean Scores')) 
print("")

print("-----------------------------------------------------")
print("----------------- Saving Outputs --------------------")
print("-----------------------------------------------------")
print("")

# Export Selected Features to .CSV
df_CML = df_models.nlargest(nModel, 'Mean Scores')
df_CML.to_csv('MultiClass_AutoCML.csv', index = False)
print("")

print("-----------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ---------")
print("---------- Signature: Mohammad Reza Saraei ----------")
print("-----------------------------------------------------")