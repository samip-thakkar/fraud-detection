# -*- coding: utf-8 -*-
"""

@author: Samip
@modified by: Tirth
"""

class Classifier:
    """ Neural Network"""

    def neural_net(x_train, y_train):
        # imports
        import keras
        import tensorflow
        from keras.models import Sequential
        from keras.layers import Dense
        # adding layers
        model = Sequential()
        model.add(Dense(64, input_dim=12, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation="sigmoid"))
        # compiling model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10)
        return model
        
    """Logistic Regression""" 
    def logistic_regression(self, x_train, y_train):
        print("Logistic Regression")
        #from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C = 1, class_weight = {1: 0.81, 0: 0.1}, penalty = 'l1', solver = 'liblinear')
        model.fit(x_train, y_train)
        return model

    """Decision Tree Classifier"""
    def decision_tree_classifier(self, x_train, y_train):
        print("Decision tree Classifier")
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
        model.fit(x_train, y_train)
        return model
    
    """KNN"""
    def knn(self, x_train, y_train):
        print("K nearest Neighbor")
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors = 5, p = 2)
        model.fit(x_train, y_train)
        return model    

    """Naive Bayes"""
    def naive_bayes(self, x_train, y_train):
        print("Naive Bayes")
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(x_train, y_train)
        return model
    
    """Random Forest"""
    def random_forest(self, x_train, y_train):
        print("Random Forrest")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators = 15, max_depth = 19, criterion = "entropy", random_state = 0, min_samples_split = 20)
        model.fit(x_train, y_train)
        return model
    
    """SVC Radial"""
    def svm(self, x_train, y_train):
        print("SVM with Radial function kernel")
        #Fitting SVM model with Radial Basis Function kernel
        from sklearn.svm import SVC
        model = SVC(kernel = "rbf", random_state = 0)
        model.fit(x_train, y_train)
        return model

    """XG-BOOST"""
    def xg_boost(self, x_train, y_train):
        print("XG-Boost")
        from xgboost import XGBClassifier
        """model = XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400, 
                                        objective="binary:hinge", booster='gbtree', 
                                        n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
                                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
                                        scale_pos_weight=1, base_score=0.5, random_state=42)"""
        model = XGBClassifier(colsample_bytree = 0.7, eta = 0.05, gamma = 0.4, max_depth = 15, min_child_weight = 1)
        model.fit(x_train, y_train)
        return model

    """ADA Boost Classification"""
    def ada_boost(self, x_train, y_train):
        print("Adaptive Boosting Classification")
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=4, random_state=0, algorithm='SAMME')
        model.fit(x_train, y_train)
        return model
        
    """LDA"""
    def lda(self, x_train, y_train):
        print("Linear Disciminant Analysis")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
        model.fit(x_train, y_train)
        return model
    
    """Neural Network"""
    def neural_net(self, x_train, y_train):
        print("Neural Network")
        # imports
        from keras.models import Sequential
        from keras.layers import Dense
        # adding layers
        model = Sequential()
        model.add(Dense(32, input_dim= x_train.shape[1], activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation="sigmoid"))
        # compiling model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, class_weight = {0:1, 1: 100})
        return model

    
