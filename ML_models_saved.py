#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from matplotlib.legend_handler import HandlerLine2D
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle




# Helper functions to add network features to input dataframe 
def add_degree(x):
    return graph_data[x]['degree']
def add_community(x):
    return str(graph_data[x]['community']) # cast to string for one-hot encoding
def add_pagerank(x):
    return graph_data[x]['pagerank']




def run_model(cid, mid, model):

  cid = "customer_'" + cid + "'"
  mid = "merchant'" + mid + "'"

  features_enhanced = pd.read_csv("data/test_data1.csv")
  cid_cols = features_enhanced[cid] == 1
  mid_cols = features_enhanced[mid] == 1

  features_enhanced = features_enhanced[cid_cols & mid_cols]
  label = features_enhanced.fraud
  features_enhanced = features_enhanced.drop('fraud', axis = 1)
  x_test = features_enhanced.to_numpy()
  y_test = label

  if(model == 'lr'):
    logisticReg(x_test, y_test)
  elif(model == 'svm'):
    SVM(x_test, y_test)
  elif(model == 'rf'):
    RF(x_test, y_test)    




def logisticReg(x_test, y_test):
  """Logistic Regression""" 
  print("Logistic Regression")
  #from sklearn.model_selection import StratifiedKFold

  filename = "models/lr_model.sav"
  lr = pickle.load(open(filename, 'rb'))
  y_pred = lr.predict(x_test)
  #Calculating accuracy by confusion matrix
  cm = metrics.confusion_matrix(y_test, y_pred)
  print(cm)
  acc = metrics.accuracy_score(y_test, y_pred)
  print("Accuracy score: ", acc)
  #Creating a classification report
  cr = classification_report(y_test, y_pred)
  print(cr)


  #filename = "D:/College/Sem4/CSE573- SWM/Project/UI/models"
  #pickle.dump(lr, open(filename, 'wb'))



def SGD():
  """SGD Classifier"""
  print("SGD Classifier")
  #from sklearn.model_selection import StratifiedKFold

  sgd = SGDClassifier()
  sgd.fit(x_train, y_train)
  y_pred = sgd.predict(x_test)
  #Calculating accuracy by confusion matrix
  cm = metrics.confusion_matrix(y_test, y_pred)
  print(cm)
  acc = metrics.accuracy_score(y_test, y_pred)
  print("Accuracy score: ", acc)
  #Creating a classification report
  cr = classification_report(y_test, y_pred)
  print(cr)


def decisionTree():
  """Decision Tree Classifier"""
  print("Decision tree Classifier")

  dtc = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
  dtc.fit(x_train, y_train)
  #Predict for x_test
  y_pred = dtc.predict(x_test)
  #Checking accuracy of Naive Bayes
  cm = metrics.confusion_matrix(y_test, y_pred) 
  print(cm)
  acc = metrics.accuracy_score(y_test, y_pred) 
  print("Accuracy score:",acc)
  #Creating a classification report
  cr = classification_report(y_test, y_pred)
  print(cr)


def KNN():
  """KNN"""
  print("K nearest Neighbor")
  #Fitting KNN classifier with L2 norm (Euclideon Distance)

  # Creating odd list K for KNN
  neighbors = list(range(1,50,2))
  # empty list that will hold cv scores
  cv_scores = [ ]
  #perform 10-fold cross-validation
  for K in neighbors:
      knn = KNeighborsClassifier(n_neighbors = K)
      scores = cross_val_score(knn,x_train,y_train,cv = 10,scoring = "accuracy")
      cv_scores.append(scores.mean())
  # Changing to mis classification error
  mse = [1-x for x in cv_scores]
  # determing best k
  optimal_k = neighbors[mse.index(min(mse))]
  print("The optimal no. of neighbors is {}".format(optimal_k))
  def plot_accuracy(knn_list_scores):
      pd.DataFrame({"K":[i for i in range(1,50,2)], "Accuracy":knn_list_scores}).set_index("K").plot.bar(figsize= (9,6),ylim=(0.78,0.83),rot=0)
      plt.show()
  plot_accuracy(cv_scores)
  knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
  knn.fit(x_train, y_train)
  #Make prediction
  y_pred = knn.predict(x_test)
  #Calculating accuracy
  cm = metrics.confusion_matrix(y_test, y_pred)
  print(cm)
  acc = metrics.accuracy_score(y_test, y_pred)
  print("Accuracy score: ", acc)
  #Creating a classification report
  cr = classification_report(y_test, y_pred)
  print(cr)


def naiveBayes():

  """Naive Bayes"""
  print("Naive Bayes")
  #Fit the model to Naive Bayes classifier


  nb = GaussianNB()
  nb.fit(x_train, y_train)
  #Predict values for x_test
  y_pred = nb.predict(x_test)
  #Checking accuracy of Naive Bayes
  cm = metrics.confusion_matrix(y_test, y_pred) 
  print(cm)
  acc = metrics.accuracy_score(y_test, y_pred) 
  print("Accuracy score:",acc)
  #Creating a classification report
  cr = classification_report(y_test, y_pred)
  print(cr)


def RF(x_test, y_test):
  """Random Forest"""
  print("Random Forrest")
  filename = "models/rf_model.sav"
  # pickle.dump(rfc, open(filename, 'wb'))
  rfc = pickle.load(open(filename, 'rb'))
  #Predict for x_test
  y_pred = rfc.predict(x_test)
  #Checking accuracy of Naive Bayes
  cm = metrics.confusion_matrix(y_test, y_pred) 
  print(cm)
  acc = metrics.accuracy_score(y_test, y_pred) 
  print("Accuracy score:",acc)
  #Creating a classification report
  cr = classification_report(y_test, y_pred)
  print(cr)


def SVM(x_test, y_test):
  """SVC Linear"""
  print("SVM with linear kernel")

  svr = SVC(kernel = "linear", random_state = 0)
  svr.fit(x_train, y_train)
  #Making predictions
  y_pred = svr.predict(x_test)
  #Checking Accuracy for x_test
  cm = metrics.confusion_matrix(y_test, y_pred)
  print("Confusion Matrix: ", cm)
  acc = metrics.accuracy_score(y_test, y_pred)
  print("Accuracy: ", acc)
  #Creating a classification report
  cr = classification_report(y_test, y_pred)
  print(cr)


  """SVC Radial"""
  print("SVM with Radial function kernel")
  #Fitting SVM model with Radial Basis Function kernel

  svr = SVC(kernel = "rbf", random_state = 0)
  svr.fit(x_train, y_train)
  #Making predictions
  y_pred = svr.predict(x_test)
  #Checking Accuracy for x_test
  cm = metrics.confusion_matrix(y_test, y_pred)
  print("Confusion Matrix: ", cm)
  acc = metrics.accuracy_score(y_test, y_pred)
  print("Accuracy: ", acc)
  #Creating a classification report
  cr = classification_report(y_test, y_pred)
  print(cr)

