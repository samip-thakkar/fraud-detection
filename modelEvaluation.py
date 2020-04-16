# -*- coding: utf-8 -*-
"""

@author: Samip
"""
from sklearn import metrics
import matplotlib.pyplot as plt
import time
class ModelEvaluation():
    
    
    def modelevaluation(self, y_test, y_pred):
        confusion = metrics.confusion_matrix(y_test, y_pred)
        print("Confussion matrix: \n", confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        print("Specificity: ", TN / (TN + FP))  
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
        print("Balanced Accuracy: ", metrics.balanced_accuracy_score(y_test, y_pred, sample_weight = None)) #Average of label accuracies
        print("Precision: ", metrics.precision_score(y_test, y_pred))
        print("Recall: ", metrics.recall_score(y_test, y_pred))
        print("F1 score macro: ",metrics.f1_score(y_test, y_pred, average='macro'))     
        print("F1 score micro: ",metrics.f1_score(y_test, y_pred, average='micro'))
        print("F-Beta score: ", metrics.fbeta_score(y_test, y_pred, beta = 10))
        print("AUC Score: ", metrics.roc_auc_score(y_test, y_pred))
        print("Zero_one_loss", metrics.zero_one_loss(y_test, y_pred))
        print("Matthews_corrcoef", metrics.matthews_corrcoef(y_test, y_pred)) #Gives equal weight to all TP, TN, FP, FN (Better than F1-score)
        print("Brier score: ", metrics.brier_score_loss(y_test, y_pred))    #The Brier score is calculated as the mean squared error between the expected probabilities for the positive class (e.g. 1.0) and the predicted probabilities. (Better than log_loss)
        print("Cohen keppa score: ", metrics.cohen_kappa_score(y_test, y_pred))     #It basically tells you how much better your classifier is performing over the performance of a classifier that simply guesses at random according to the frequency of each class.
        print("Classification_report\n", metrics.classification_report(y_test, y_pred))
        
        
        
    def plotROC(self, y_test, y_pred):        #Receiver Operating Characteristic (ROC)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
        print("AUC Score: ", metrics.roc_auc_score(y_test, y_pred))
        
        
    def plotPrecisionRecall(self, y_test, y_pred):
        #average_precision = metrics.average_precision_score(y_pred, y_test)
        #print('Average precision-recall score: {}'.format(average_precision))
        time.sleep(5)
        precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred)
        print("AUC for Precision Recall: ", metrics.auc(precision, recall))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

        
    def crossValScore(self, model, X, y):
        from sklearn.model_selection import cross_val_score
        print("Cross validation score: ", cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean())
        