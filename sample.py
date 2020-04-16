# -*- coding: utf-8 -*-
"""

@author: Samip
"""
import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, NearMiss, CondensedNearestNeighbour, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek


class Sample:        
    
    """Random Under sampling"""
    def randomundersample(self, x_train, y_train):
        rus = RandomUnderSampler()
        X_rus, y_rus = rus.fit_sample(x_train, y_train)
        return X_rus, y_rus
    
    """Random Over sampling"""
    def randomoversample(self, x_train, y_train):
        ros = RandomOverSampler()
        X_ros, y_ros = ros.fit_sample(x_train, y_train)
        return X_ros, y_ros


    """Under Sampling Tomek Links
    def tomekundersample(self, x_train, y_train):
        tl = TomekLinks()
        X_tl, y_tl, id_tl = tl.fit_sample(x_train, y_train)
        return X_tl, y_tl"""
    
    """Under-sampling: Cluster Centroids"""
    def clustercentroidundersample(self, x_train, y_train):
        cc = ClusterCentroids()
        X_cc, y_cc = cc.fit_sample(x_train, y_train)
        return X_cc, y_cc
    
    """Near Miss
    def nearmissundersample(self, x_train, y_train):
        nm = NearMiss()
        X_nm, y_nm = nm.fit_sample(x_train, y_train)
        return X_nm, y_nm"""
    
    """Condensed Nearest Neighbour
    def cnnundersample(self, x_train, y_train):
        cnn = CondensedNearestNeighbour(n_neighbors = 1)
        X_cnn, y_cnn = cnn.fit_sample(x_train, y_train)
        return X_cnn, y_cnn"""
    
    """Edited Nearest Neighbour
    def ennundersample(self, x_train, y_train):
        enn = EditedNearestNeighbours(n_neighbors = 3)
        X_enn, y_enn = enn.fit_sample(x_train, y_train)
        return X_enn, y_enn"""
    
    """SMOTE"""
    def smote(self, x_train, y_train):
        smt = SMOTE()
        x_resample, y_resample = smt.fit_sample(x_train, y_train)
        return x_resample, y_resample
    
    """Over-sampling followed by under-sampling
    def combine(self, x_train, y_train):
        smt = SMOTETomek()
        X_smt, y_smt = smt.fit_sample(x_train, y_train)
        return X_smt, y_smt"""