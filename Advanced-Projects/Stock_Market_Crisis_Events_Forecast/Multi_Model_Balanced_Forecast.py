# -*- coding: utf-8 -*-
"""
@author: Ludovic de Villelongue
"""
#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy import stats
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import random as random
from sklearn import tree
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import statsmodels.api as sm    

#Mute sklearn warnings
import warnings
warnings.filterwarnings("ignore")


#Execution time 
import time
StartTime=time.time()

'''
-----------------------------------------
------------Data Processing--------------
-----------------------------------------
'''
df=pd.read_csv(r'C:/Users/ludov/Documents/Dauphine/M2/S1/Gestion Quantitative/projet/bonds_dataset.csv',index_col=0)

df_bs=pd.read_csv(r'C:/Users/ludov/Documents/Dauphine/M2/S1/Gestion Quantitative/projet/bonds_dataset.csv',index_col=0)

def clean_dataset(df_c):
    assert isinstance(df_c, pd.DataFrame), "df needs to be a pd.DataFrame"
    df_c.dropna(inplace=True)
    indices_to_keep = ~df_c.isin([np.nan, np.inf, -np.inf]).any(1)
    return df_c[indices_to_keep].astype(np.float64)

df_clean=clean_dataset(df)
df_clean_bs=clean_dataset(df_bs)


def balance_train(df_sel):
    
    #Dataset split 
    no_crash, crash = df_clean[df_clean.iloc[:, -1]==0], df_clean[df_clean.iloc[:, -1]==1]
    l_no_crash, l_crash = len(no_crash), len(crash)
    
    #train balance
    crash_train = crash.sample(int(l_no_crash*.75),replace=True)
    no_crash_train = no_crash.sample(int(l_no_crash*.75))
    Train = pd.concat([crash_train, no_crash_train], axis=0)
    return Train

def balance_test(df_sel):
    #Dataset split 
    no_crash, crash = df_clean[df_clean.iloc[:, -1]==0], df_clean[df_clean.iloc[:, -1]==1]
    l_no_crash, l_crash = len(no_crash), len(crash)
    #test balance
    crash_test = crash.sample(int(l_no_crash*.25),replace=True)
    no_crash_test = no_crash.sample(int(l_no_crash*.25))
    Test = pd.concat([crash_test, no_crash_test], axis=0)
    return Test

trainset=balance_train(df_clean)
testset=balance_test(df_clean)
trainset_bs=balance_train(df_clean_bs)
testset_bs=balance_test(df_clean_bs)
# Extract labels
Ytrain = trainset.iloc[:,-1]
Ytest = testset.iloc[:,-1]
Ytrain_bs = trainset_bs.iloc[:,-1]
Ytest_bs=testset_bs.iloc[:,-1]
# Extract features
Xtrain = trainset.iloc[:,:-1] 
Xtest = testset.iloc[:,:-1]
Xtrain_bs=trainset_bs.iloc[:,:-1]
Xtest_bs=testset_bs.iloc[:, :-1]
    



'''
difference in features value magnitudes
'''

from sklearn import preprocessing

def norm(df_select):

    scaler = preprocessing.MinMaxScaler()
    names = df_select.columns
    d  = scaler.fit_transform(df_select)
    scaled_df = pd.DataFrame(d, columns=names)
    return scaled_df

Xtrain_norm= norm(Xtrain)
Xtest_norm = norm(Xtest)



'''
-----------------------------------------
---------------Algorithms----------------
-----------------------------------------
'''



#Feature importance
def feature_importance(classifier,classifier_name, X_actual):
    #MDI feature importance
    features = X_actual.columns
    importances = classifier.feature_importances_
    indices = np.argsort(importances)
    indices=indices[:15]
    plt.figure()
    plt.title(' {} Feature Importances (Mean Decrease in Impurity)'.format(classifier_name))
    plt.barh(features[indices], importances[indices], align='center')
    plt.xlabel('Relative Importance')
    plt.show()
    
    #Permutation feature importance
    plt.figure()
    perm_importance = permutation_importance(classifier, Xtrain_norm, Ytrain, \
                                             scoring='neg_mean_squared_error')
    sorted_idx = perm_importance.importances_mean.argsort()
    sorted_idx=sorted_idx[:15]
    plt.title(' {} Feature Importances (Permutation MSE)'.format(classifier_name))
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importances")
    plt.show()

# '''
# GLM Logistic Regression
# '''
# #Logistic regression feature selection
# log_glm=sm.GLM(Ytrain,(sm.add_constant(Xtrain)), family = sm.families.Binomial()).fit()
# print(log_glm.summary(), "\n")
# # # '''only take variables with p-values under 15%''' Xtrain_norm.drop('ticker name',axis=1, inplace=True)
# # Xtrain_norm_select=Xtrain_norm

'''
Elastic net penalized logistic regression
'''
sgd_clf_pen = SGDClassifier(random_state=42, loss='log', shuffle=True, penalty='elasticnet', l1_ratio=0.5)
sgd_clf_pen.fit(Xtrain_norm, Ytrain)


'''
Optimal penalized logistic regression
'''
#Grid search Lasso Regression
param_grid = {'l1_ratio': np.arange(0,1.1,.1)}
grid_search = GridSearchCV(sgd_clf_pen, param_grid, cv=5, scoring='roc_auc')
x = grid_search.fit(Xtrain, Ytrain)
x.cv_results_
x.best_params_
sgd_clf_pen_optim = x.best_estimator_
sgd_clf_pen_optim.fit(Xtrain_norm, Ytrain)


'''
Broyden–Fletcher–Goldfarb–Shanno Logistic Regression
'''
Log_clf = LogisticRegression(solver="lbfgs") # Broyden–Fletcher–Goldfarb–Shanno algorithm
Log_clf.fit(Xtrain_norm, Ytrain)


'''
Max deicision tree
'''
DTC_max=DecisionTreeClassifier()
DTC_max.fit(Xtrain_norm,Ytrain)


'''
Optim decision tree
'''
DTC=DecisionTreeClassifier()
#Grid search for depth and leaf
param_grid = {'max_depth': np.arange(3,8), 'min_samples_leaf' : np.arange(5,20)}
grid_search = GridSearchCV(DTC, param_grid, cv=5, scoring='roc_auc')
x = grid_search.fit(Xtrain_norm, Ytrain)
x.best_params_
DTC_optim=x.best_estimator_
DTC_optim.fit(Xtrain_norm,Ytrain)
tree.plot_tree(DTC_optim)

'''
Max Random Forest
'''
RFC_max = RandomForestClassifier(random_state=42)
RFC_max.fit(Xtrain_norm, Ytrain)


'''
Optim Random Forest
'''
    
RFC=RandomForestClassifier()
#Grid search for number of trees and number of splits
param_grid = {'n_estimators' : np.arange(5,20),
              'max_depth': np.arange(5,15)
             }
grid_search = GridSearchCV(RFC, param_grid, cv=5, scoring='roc_auc', n_jobs=8)
x = grid_search.fit(Xtrain, Ytrain)
x.best_params_
RFC_optim=x.best_estimator_
RFC_optim.fit(Xtrain_norm,Ytrain)
features=feature_importance(RFC_optim, "Random Forest", Xtest_norm)



'''
Optimal SVM
'''
#Grid search for optimal kernel, c and gamma
param_grid = {'kernel': ['rbf','polynomial','sigmoid'],
              'C': [0.1, 1, 10, 100],
               'gamma': [1, 0.1, 0.01, 0.001]}
              
grid_search = GridSearchCV(SVC(probability=True),\
                           param_grid,refit = True, cv=5, verbose=False)
x = grid_search.fit(Xtrain, Ytrain)
x.cv_results_
x.best_params_
SVC_optim = x.best_estimator_
SVC_optim.fit(Xtrain_norm, Ytrain)



'''
 Neural Network
'''
#Function to generate Deep ANN model 
def make_classification_ann(Optimizer_Trial, Neurons_Trial):
    
    #Classifier ANN model
    classifier = Sequential()
    classifier.add(Dense(units=Neurons_Trial, input_dim=9, kernel_initializer='uniform', activation='sigmoid'))
    classifier.add(Dense(units=Neurons_Trial, kernel_initializer='uniform', activation='sigmoid'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=Optimizer_Trial, loss='binary_crossentropy', metrics=['accuracy'])
            
    return classifier

 
Parameter_Trials={'batch_size':np.arange(10,20,10),
                      'epochs':np.arange(10,20,10),
                    'Optimizer_Trial':['adam', 'rmsprop'],
                  'Neurons_Trial': np.arange(1,11,1)
                 }
 
#Classifier ANN
classifierModel=KerasClassifier(make_classification_ann)

#Grid search space
grid_search=GridSearchCV(estimator=classifierModel, param_grid=Parameter_Trials,\
                         scoring='roc_auc', cv=5,verbose=False)
x.cv_results_
x.best_params_
neural_net_optim = x.best_estimator_
neural_net_optim.fit(Xtrain_norm, Ytrain)
 


'''
XGBoost
'''

#Deprecated functions warning desactivated by using use_label_encoder=False
XGb = xgb.XGBClassifier(objective='binary:logistic',eval_metric="error", use_label_encoder =False )

#Grid search for maximum depth of generated tree, minium leaf node size,
#subsampling for building classification trees, subsampling for variable considered in each split
param_grid = {
    'max_depth': np.arange(5,6),
    'min_child_weight':np.arange(2,3),
    'colsample_bytree': np.arange(0.1,1,0.2),
    'colsample_bynode': np.arange(0.1,1,0.2)
}
grid_search = GridSearchCV(XGb, param_grid, cv=5, scoring='roc_auc')
x = grid_search.fit(Xtrain, Ytrain, eval_metric="auc",\
                    eval_set=[(Xtest, Ytest)], early_stopping_rounds=2,verbose=False)
x.cv_results_
x.best_params_
XGb_optim = x.best_estimator_
XGb_optim.fit(Xtrain_norm, Ytrain)
features=feature_importance(XGb_optim, "XGBoost", Xtest_norm)



'''
MLP neural networks
'''

#Initializing the Recurrent Neural Network
neural_net = MLPClassifier(alpha=1e-5, random_state=1)
#Grid search for number of hidden layers and number of units
param_grid = {'hidden_layer_sizes': [(random.randint(1, 10), random.randint(1, 10)) for i in range(0, 10)],
              'activation': ["logistic", "relu", "Tanh"],
              'solver':['lbfgs', 'adam', 'sgd']}
grid_search = GridSearchCV(estimator=neural_net, param_grid=param_grid, cv=5,\
                           verbose=False)
x = grid_search.fit(Xtrain, Ytrain)
x.cv_results_
x.best_params_
deep_neural_net_optim = x.best_estimator_
deep_neural_net_optim.fit(Xtrain_norm, Ytrain)



'''
-----------------------------------------
------------Evaluation Process-----------
-----------------------------------------
'''

'''
ROC Curve Plot
'''

def plot_roc_curve(fpr, tpr, color):
    plt.plot(fpr, tpr, color,  linewidth=2)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

'''
Evaluation Steps
'''

def evaluation(X_Actual, Y_actual, classifier, classifier_name):
    Y_predicted = classifier.predict(X_Actual)
    cm = confusion_matrix(Y_actual, Y_predicted)
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=classifier.classes_).plot()
    plt.title(classifier_name)

    #5 fold cross validation
    x = cross_val_score(classifier, X_Actual, Y_actual, cv=5)
    print("cross validation score:", x, "\n")
    Yscores = cross_val_predict(classifier, X_Actual, Y_actual, cv=5,\
                                method="predict_proba")
    Yscores=Yscores[:,-1]

    #ROC curve
    fpr, tpr, thresholds = roc_curve(Y_actual, Yscores)
    
    #Metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity=tp/(tp+fn)
    fpr_cm=fp/(fp+tn)
    specificity=tn/(tn+fp)
    G=math.sqrt(sensitivity*specificity)
    LP=sensitivity/(1-specificity)
    LR=(1-sensitivity)/specificity
    #DP=(math.sqrt(3)/math.pi)*(math.log(sensitivity/(1-sensitivity))+math.log(specificity/(1-specificity)))
    gamma=sensitivity-(1-sensitivity)
    BA=(1/2)*(sensitivity+specificity)
    WBA=(3/4)*(sensitivity)+(1/4)*specificity
    index_metrics = ["false alarm","hit rate","precision","accuracy","recall", \
                "AUC","Kolmogorov–Smirnov","G-mean","Positive Likelihood Ratio",\
                  "Negative Likelihood Ratio", "Youden Index",\
                      "Balanced Accuracy", "Weighted Balance Accuracy"]
    d = {classifier_name: [fpr_cm, sensitivity, precision_score(Y_actual, Y_predicted)\
                           ,accuracy_score(Y_actual, Y_predicted),\
                               recall_score(Y_actual, Y_predicted),\
                                   roc_auc_score(Y_actual, Yscores),\
                           stats.ks_2samp(Y_actual, Yscores)[0],G,LP,\
                               LR,gamma,BA,WBA]}
    df_metrics=pd.DataFrame(d, index_metrics)
    return fpr, tpr, classifier_name, df_metrics




'''
Evaluation per model
'''
def train_test(algo, train_name,test_name): #rajouter X_actu, y actu, x actu test, y actu test qui prendra les variables notées a 4/4 ou Xtrain_norm...
    train_evaluation=evaluation(Xtrain_norm, Ytrain, algo, train_name)
    test_evaluation=evaluation(Xtest_norm, Ytest, algo,test_name)
    return train_evaluation, test_evaluation

#Algo List
algo_1=train_test(sgd_clf_pen, "Optimal Penalized Logistic Regression train",\
                  "Optimal Penalized Logistic Regression test")
algo_2=train_test(sgd_clf_pen_optim, "elastic net Penalized Logistic Regression train",\
                  "elastic net Penalized Logistic Regression test")
algo_3=train_test(Log_clf,"Lbfgs Logistic Regression train" ,\
                  "Lbfgs Logistic Regression test" )
algo_4=train_test(SVC_optim, "Optimal SVM train", "Optimal SVM test")
algo_5=train_test(DTC_max, "Max Decision Tree train", "Max Decision Tree test")
algo_6=train_test(DTC_optim, "Optim Decision Tree train", "Optim Decision Tree test")
algo_7=train_test(RFC_max, "Max Random Forest train", "Max Random Forest test")
algo_8=train_test(RFC_optim, "Optim Random Forest train", "Optim Random Forest test")
algo_9=train_test(neural_net_optim, "Neural Network train", "Neural Network test")
algo_10=train_test(XGb_optim, "XGBoost train", "XGBoost test")
algo_11=train_test(deep_neural_net_optim,"MLP Neural Network train",\
                   "MLP Neural Network test")

'''
Display ROC Curve and create metrics comparison dataframe
'''


def metrics_total(model_1, model_2, model_3, model_4, model_5, model_6, model_7,
                  model_8,model_9,model_10, model_11):
    #Dataframe metrics
    df_total_metrics = pd.concat([model_1[3], model_2[3], model_3[3],
                                 model_4[3], model_5[3], model_6[3],
                                  model_7[3], model_8[3], model_9[3],
                                 model_10[3], model_11[3]], axis=1)
    #Comparative ROC Curve
    plt.figure(figsize=(8, 6))
    plot_roc_curve(model_1[0], model_1[1], 'g-')
    plot_roc_curve(model_2[0], model_2[1], 'b-')
    plot_roc_curve(model_3[0], model_3[1],'r-')
    plot_roc_curve(model_4[0], model_4[1],'c-')
    plot_roc_curve(model_5[0], model_5[1],'m-')
    plot_roc_curve(model_6[0], model_6[1],'y-')
    plot_roc_curve(model_7[0], model_7[1],'k-')
    plot_roc_curve(model_8[0], model_8[1],'g--')
    plot_roc_curve(model_9[0], model_9[1],'b--')
    plot_roc_curve(model_10[0], model_10[1],'r--')
    plot_roc_curve(model_11[0], model_11[1],'c--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.legend([model_1[2], model_2[2],model_3[2], model_4[2], model_5[2],\
                model_6[2], model_7[2],model_8[2],model_9[2],model_10[2], \
                   model_11[2]],bbox_to_anchor=(1.04, 1))
    plt.show()
    return df_total_metrics

df_train_metrics=metrics_total(algo_1[0], algo_2[0],algo_3[0],algo_4[0],\
                               algo_5[0], algo_6[0],algo_7[0],algo_8[0],\
                                   algo_9[0],algo_10[0], algo_11[0])
df_test_metrics=metrics_total(algo_1[1], algo_2[1],algo_3[1],algo_4[1],\
                              algo_5[1],algo_6[1],algo_7[1],algo_8[1],\
                                   algo_9[1],algo_10[1], algo_11[1])
df_train_metrics.to_csv(r'/Users/ludov/Documents/Dauphine/M2/S1/Gestion Quantitative/dataset_bd.csv', index = True)
df_test_metrics.to_csv(r'/Users/ludov/Documents/Dauphine/M2/S1/Gestion Quantitative/dataset_tt_bd.csv', index = True)

#Execution time
EndTime=time.time()
print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')
