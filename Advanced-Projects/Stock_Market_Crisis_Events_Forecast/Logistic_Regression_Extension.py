# -*- coding: utf-8 -*-
"""
@author: Ludovic de Villelongue
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import SGDClassifier

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

'''
difference in features value magnitudes
'''


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
Standard Logistic Regression with sklearn
'''

#Broyden–Fletcher–Goldfarb–Shanno algorithm
Log_clf = LogisticRegression(solver="lbfgs")
Log_clf.fit(Xtrain_norm, Ytrain)
Ypred = Log_clf.predict(Xtrain_norm)
print("accuracy_Logistic_Regression:{:.2f}%".format(100*accuracy_score(Ytrain, Ypred)))



'''
Stochastic Gradient Descent Without sklearn

'''
'''

- Write function `sigmo(t)` that returns the value of the sigmoid value at $t$, i.e.:
$$\sigma(t) = 1/(1+\exp(-t))$$

- Write the function `sgd_gradw(x, y, w, b)` that reports the value of the gradient components with respect to `w` for a single instance, i.e.:
$$x^{(i)}\left[\sigma\left(w^Tx^{(i)}+b\right) - y^{(i)} \right]$$

- Write the function `sgd_gradb(x, y, w, b)` that reports the value of the partial derivative of the cost function with respect to `b` for a single instance, i.e.:
$$\sigma\left(w^Tx^{(i)}+b\right) - y^{(i)}$$

- Write the function `cost(x, y, w, b)` that returns the costs for instance $i$, i.e.:
$$-y^{(i)} \log \sigma
\left(w^Tx^{(i)}+b\right) - \left(1 - y^{(i)}\right)
\log\left[1 - \sigma
\left(w^Tx^{(i)}+b\right) \right]$$

- Write the function `prediction(x, w, b)`that returns the predicted $\hat{y}^{(i)}$

'''


def sigmo(t):
    return 1/(1+np.exp(-t))

def sgd_gradw(x, y, w, b):
    return x*(sigmo(np.dot(x,w)+b)-y)

def sgd_gradb(x, y, w, b):
    return sigmo(np.dot(x,w)+b)-y

def cost(x, y, w, b):
    return -y*np.log(sigmo(np.dot(x,w)+b)) - (1-y)*np.log(1-sigmo(np.dot(x,w)+b))

def prediction(x, w, b):
    out = sigmo(np.dot(x,w)+b)
    if out >= .5:
        return 1
    else:
        return 0
    
eta = 0.01
loss_list=[]
np.random.seed(42)
#Random initialization of w and b
w = np.random.uniform(low=-1, high=1, size=Xtrain_norm.shape[1]) 
b = np.random.uniform(low=-1, high=1, size=1)

for epoch in range(100):
    #Shuffle randomly Xtrain_norm
    Xtrain_rand = Xtrain_norm.sample(frac=1)
    #Reorder Ytrain in the same order as Xtrain_rand
    Ytrain_rand = Ytrain.reindex(Xtrain_rand.index)
    for i in range(Xtrain_rand.shape[0]):
        x = Xtrain_rand.iloc[i,:]
        y = Ytrain_rand.iloc[i]
        grad_w = sgd_gradw(x, y, w, b)
        grad_b = sgd_gradb(x, y, w, b)
        w -=eta * grad_w
        b -=eta * grad_b
    loss = 0
    for i in range(Xtrain_rand.shape[0]):
        x = Xtrain_rand.iloc[i,:]
        y = Ytrain_rand.iloc[i]
        loss += cost(x, y, w, b)
    loss = loss / Xtrain_rand.shape[0]
    loss_list.append(loss)
    if epoch % 10 == 0:
        print("loss epoch", epoch, ":", loss)
print("loss epoch", epoch, ":", loss)
print("\nfinished\n")
epoch = np.arange(len(loss_list))
plt.plot(epoch, loss_list)
count_ok = 0
for i in range(Xtrain_norm.shape[0]):
        x = Xtrain_norm.iloc[i,:]
        y = Ytrain.iloc[i]
        if prediction(x, w, b) == y:
            count_ok += 1
print("accuracy_SGD_without_SKlearn:{:.2f}%".format(100 * count_ok/Xtrain_norm.shape[0]), "\n")





'''
Stochastic Gradient Descen with sklearn
'''


sgd_clf = SGDClassifier(random_state=42, loss='log', shuffle=True, penalty='none',
                        learning_rate='constant', eta0=0.01)
sgd_clf.fit(Xtrain_norm, Ytrain)
Ypred = sgd_clf.predict(Xtrain_norm)
accuracy_score(Ytrain, Ypred)
print("accuracy_SGD:{:.2f}%".format(100*accuracy_score(Ytrain, Ypred)),"\n")




'''
Mini-batch gradient descent without sklearn
'''


def sgdb_gradw(X, Y, w, b):
    m = X.shape[0]
    return np.dot((sigmo(np.dot(X,w)+b)-Y).T,X)/m
def sgdb_gradb(X, Y, w, b):
    return np.mean(sigmo(np.dot(X,w)+b)-Y)
def costb(X, Y, w, b):
    return np.dot(Y, np.log(sigmo(np.dot(X,w)+b)))+np.dot((1-Y), np.log(1-sigmo(np.dot(X,w)+b)))


eta = 0.01
loss_list=[]
np.random.seed(42)
w2 = np.random.uniform(low=-1, high=1, size=Xtrain_norm.shape[1])
b2 = np.random.uniform(low=-1, high=1, size=1)
batch_size = 5
n_iter = Xtrain_norm.shape[0]//batch_size +1
   
for epoch in range(100):
    Xtrain_rand = Xtrain_norm.sample(frac=1)                
    Ytrain_rand = Ytrain.reindex(Xtrain_rand.index)
    for i in range(n_iter):
        X = Xtrain_rand[i*batch_size:(i+1)*batch_size]                          
        Y = Ytrain_rand[i*batch_size:(i+1)*batch_size]
        grad_w2 = sgdb_gradw(X, Y, w2, b2)
        grad_b2 = sgdb_gradb(X, Y, w2, b2)
        w2 -= eta*grad_w2
        b2 -= eta*grad_b2
    loss = -costb(Xtrain_rand, Ytrain_rand, w2, b2) / Xtrain_rand.shape[0]
    loss_list.append(loss)
    if epoch % 10 == 0:
        print("loss epoch", epoch, ":", loss)
print("loss epoch", epoch, ":", loss)
print("\nfinished\n")
epoch = np.arange(len(loss_list))
plt.plot(epoch, loss_list)
#Vectorized version of prediction accuracy
def accuracy(X, Y, w, b):
    pred = (sigmo(np.dot(X, w)+b)>=0.5)
    check = (pred==Y)
    return check.sum()/len(pred)
print("accuracy_Mini_Batch_Without_SKLearn: {:.2f}%".format(100 * accuracy(Xtrain_norm, Ytrain, w2, b2)), "\n")






''' 
Confusion Matrix
'''

def print_metrics(fpr_cm, tpr_cm, Y_actual, Y_predicted, classifier_name):
    print(classifier_name)
    print("false alarm:", fpr_cm)
    print("hit rate:", tpr_cm)
    print("accuracy:", accuracy_score(Y_actual, Y_predicted))
    print("precision:", precision_score(Y_actual, Y_predicted))
    print("recall:", recall_score(Y_actual, Y_predicted), "\n")

cm = confusion_matrix(Ytrain, Ypred)
confusion_matrix=ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=sgd_clf.classes_).plot()
#True positive rate and false positive rate
tn, fp, fn, tp = cm.ravel()
tpr_cm=tp/(tp+fn)
fpr_cm=fp/(fp+tn)
print_metrics(fpr_cm, tpr_cm, Ytrain, Ypred, "sgd_clf_train")



'''
K fold Cross Validation
'''


#5 fold
scores_sgd_clf = cross_val_score(sgd_clf, Xtrain_norm, Ytrain,
                                 cv=5, scoring='precision')
print("cross validation score:", scores_sgd_clf, "\n")

'''
Area under curve
'''

Yscores = cross_val_predict(sgd_clf, Xtrain_norm, Ytrain, cv=None,
                             method="decision_function")

fpr, tpr, thresholds = roc_curve(Ytrain, Yscores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()
roc_auc_score(Ytrain,Yscores)


def print_metrics2(classifier, classifier_name):
    y_pred = classifier.predict(Xtest_norm)
    print(classifier_name)
    print("precision:", precision_score(Ytest, y_pred))
    print("accuracy:", accuracy_score(Ytest, y_pred))
    print("recall:", recall_score(Ytest, y_pred))
    print("AUC:", roc_auc_score(Ytest, y_pred),"\n")
    
print_metrics2(sgd_clf, "sgd_clf_test")





