# -*- coding: utf-8 -*-
"""
@author: Ludovic de Villelongue
"""

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
import pandas as pd

#Mute sklearn warnings
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv(r'C:/Users/ludov/Documents/Dauphine/M2/AI for economics and finance/course 7_Bankrputcy prediction part 2/trainset_2.csv')
#Dataset split into Xtrain, Ytrain, Xtest and Ytest
split = len(df) * 0.7
trainset = df.iloc[:int(split), :]
testset = df.iloc[int(split):, :]
Xtrain, Ytrain = trainset.iloc[:, :-1], trainset.iloc[:, -1]
Xtest, Ytest = testset.iloc[:, :-1], testset.iloc[:, -1]


def normalize(*arg):  # df [,min_vals,max_vals]
    """ Normalizes the features of a dataframe
    Up to 3 arguments
    - 1st argument (mandatory) = df: name of dataframe to normalize
    - 2nd argument (optional): min_vals = list of min value of each feature
    - 3rd argument (optional): max_vals = list of max value of each feature
    If df only is passed, the function normalizes X_ij, i.e. ith value of feature j as:
                               (X_ij - min_j)/(max_j - min_j)                           (1)
    and returns the min_j and max_j lists
    If df, min_vals and max_vals are passed, df is normalized as per (1) using these values
    """
    df = arg[0]
    result = df.copy()
    if len(arg) == 1:
        minval, maxval = [], []
        for feature_name in df.columns:
            min_value = df[feature_name].min()
            max_value = df[feature_name].max()
            result[feature_name] = (
                df[feature_name] - min_value) / (max_value - min_value)
            minval.append(min_value)
            maxval.append(max_value)
        return result, minval, maxval
    else:
        minvals = arg[1]
        maxvals = arg[2]
        for i in range(df.shape[1]):
            result.iloc[:, i] = (df.iloc[:, i] - minvals[i]
                                 ) / (maxvals[i] - minvals[i])
        return result


Xtrain_norm, min_vals, max_vals = normalize(Xtrain)
Xtest_norm = normalize(Xtest, min_vals, max_vals)



'''
SVM
'''

def print_metrics(fpr_cm, tpr_cm,Y_actual, Y_predicted):
    print("false alarm:", fpr_cm)
    print("hit rate:", tpr_cm)
    print("accuracy:", accuracy_score(Y_actual, Y_predicted))
    print("precision:", precision_score(Y_actual, Y_predicted))
    print("recall:", recall_score(Y_actual, Y_predicted), "\n")


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

#Linear kernel SVM
svclassifier_lin = SVC(kernel='linear')
svclassifier_lin.fit(Xtrain_norm, Ytrain)
y_pred_lin = svclassifier_lin.predict(Xtrain_norm)


#Polynomial kernel SVM
svclassifier_poly = SVC(kernel='poly', degree=2)
svclassifier_poly.fit(Xtrain_norm, Ytrain)
y_pred_poly = svclassifier_poly.predict(Xtrain_norm)

#Sigmoid kernel SVM
svclassifier_sigmo = SVC(kernel='sigmoid')
svclassifier_sigmo.fit(Xtrain_norm, Ytrain)
y_pred_sigmo = svclassifier_sigmo.predict(Xtrain_norm)



#Gaussian kernel
svclassifier_rbf = SVC(kernel='rbf')
svclassifier_rbf.fit(Xtrain_norm, Ytrain)
y_pred_rbf = svclassifier_rbf.predict(Xtrain_norm)


def evaluation(y_pred, classifier, classifier_name):
    cm = confusion_matrix(Ytrain, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=classifier.classes_).plot()

    #5 fold
    Yscores = cross_val_predict(classifier, Xtrain_norm, Ytrain, cv=None,
                             method="predict")

    fpr, tpr, thresholds = roc_curve(Ytrain, Yscores)
    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr)
    plt.show()
    roc_auc_score(Ytrain,Yscores)
    y_pred = classifier.predict(Xtest_norm)
    print(classifier_name)
    print("precision:", precision_score(Ytest, y_pred))
    print("accuracy:", accuracy_score(Ytest, y_pred))
    print("recall:", recall_score(Ytest, y_pred))
    print("AUC:", roc_auc_score(Ytest, y_pred),"\n")
    
    
evaluation(y_pred_lin,svclassifier_lin, "linear kernel SVM test")
evaluation(y_pred_poly,svclassifier_poly, "polynomial kernel SVM test")
evaluation(y_pred_sigmo,svclassifier_sigmo, "sigmoid kernel SVM test")
evaluation(y_pred_rbf,svclassifier_rbf, "gaussian kernel SVM test")