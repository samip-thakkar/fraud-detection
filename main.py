# -*- coding: utf-8 -*-
"""
@author: Samip

"""
        
from extract_graph_features import GraphFeatures
from preprocessing import Preprocess
from sample import Sample
from classifiers import Classifier
from modelEvaluation import ModelEvaluation
from preprocessed_graph import PreprocessGraph

#Create the objects
gf = GraphFeatures()
pp = Preprocess()
ppg = PreprocessGraph()
sample = Sample()
classifier = Classifier()
me = ModelEvaluation()

#Only for one time
#gf.extractGraphFeatures()

choice = int(input("Enter 0 for data without graph features and 1 for graph features: "))
#Preprocess the data
if choice == 0:
    x_train, x_test, y_train, y_test = pp.do_mca()
else:
    print("Extracting graph features: ")
    x_train, x_test, y_train, y_test = ppg.scale_data()
    print(x_train.head())

#Sampling the data
choice = input(("Do you want to sample the data(y/n)? "))
if choice == 'yes' or choice == 'y':
    choice = int(input("Enter 1 for Over sampling, 2 for Under sampling and 3 for SMOTE: "))
    d = {1: sample.randomoversample, 2: sample.randomundersample, 3:sample.smote}
    x_train, y_train = d[choice](x_train, y_train)


choice = int(input("Enter 1 for Logistic Regression, 2 for Decision Tree Classifier, 3 for KNN, 4 for Naive Bayes, 5 for Random Forest, 6 for SVM, 7 for XG Boost, 8 for Adaptive Boosting, 9 for LDA: "))
clf = {1: classifier.logistic_regression, 2: classifier.decision_tree_classifier, 3: classifier.knn, 4: classifier.naive_bayes, 5: classifier.random_forest, 6: classifier.svm, 7: classifier.xg_boost, 8: classifier.ada_boost, 9: classifier.lda, 10:classifier.neural_net}

model = clf[choice](x_train, y_train)

#Get the predicted values
y_pred = model.predict(x_test)

#Get the model evaluation
me.modelevaluation(y_test.to_numpy(), y_pred)

#Get the ROC Curve
me.plotROC(y_test, y_pred)