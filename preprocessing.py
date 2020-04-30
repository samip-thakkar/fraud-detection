# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Import libraries
import pandas as pd
import numpy as np 
from sample import Sample
from sklearn.preprocessing import LabelEncoder

sample = Sample()


class Preprocess():
    # columns = []
    # from sklearn.preprocessing import MinMaxScaler
    # min_max_scaler = MinMaxScaler()    
    
    """Read the data"""
    def read_data(self):
        df = pd.read_csv('data/bs140513_032310.csv')
        return df

    """Data Visualization"""
    def data_visualization(self):
        df = self.read_data()
        print(df.nunique())
        target_count = df.fraud.value_counts()
        print('Class 0:', target_count[0])
        print('Class 1:', target_count[1])
        print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
        #target_count.plot(kind='bar', title='Count (target)')
        return df
    
    """Data Cleaning"""
    def data_cleaning(self):
        df = self.data_visualization()
        #Drop unnecessary columns
        df = df.drop(['zipcodeOri', 'zipMerchant'], axis = 1)
        
        #Clean the data
        #df['age'] = df['age'].apply(lambda x: x[1]).replace('U', 7).astype(int)
        #df['gender'] = df['gender'].apply(lambda x: x[1])
        df['merchant'] = df['merchant'].apply(lambda x: x[1:-1])
        #df['category'] = df['category'].apply(lambda x: x[1:-1])
        df['customer'] = df['customer'].apply(lambda x: x[2:-1]).astype(float)
        #Random shuffle
        df = df.sample(frac = 1).reset_index(drop = True)
        return df
    
    """Data Transformation"""
    def data_transformation(self):
        df = self.data_cleaning()
        #Describe the data as features and target class as label
        features = df.drop('fraud', axis = 1)
        label = df.fraud
        features = features[['customer','amount', 'merchant']]
        # Get validation data
        features = pd.get_dummies(features, columns = ['merchant'])
        #Label Encoding the data
        return features, label

    
    """Split the data for training and testing data"""
    def split_data(self):
        features, label = self.data_transformation()
        #Splitting data into training data and testing data
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(features, label, train_size = 0.8, random_state = 42, stratify = label)
        # self.columns = features.columns
        # self.columns = self.columns.insert(self.columns.shape[0], 'fraud')      
        return x_train, x_test, y_train, y_test
        
    """Scale the data to (0,1) as higher range values might overpower the smaller range during the calculation"""
    def scale_data(self):
        x_train, x_test, y_train, y_test = self.split_data()
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler()
        x_train['amount'] = min_max_scaler.fit_transform(x_train['amount'].values.reshape(-1, 1))
        x_test['amount'] = min_max_scaler.fit_transform(x_test['amount'].values.reshape(-1, 1))
        return x_train, x_test, y_train, y_test
    
    
    """Return final preprocessed data"""
    def preprocessed_data(self):
        x_train, x_test, y_train, y_test = self.scale_data()        
        return x_train, x_test, y_train, y_test

    """Return final preprocessed data"""
    def preprocess_data_ui(self):
        x_train, x_test, y_train, y_test = self.split_data() 

        # Sampling after preprocessing
        # x_train, y_train = sample.smote(x_train, y_train)

        x_train.to_csv("data/train/x_train.csv",index=False)
        x_test.to_csv("data/test/x_test.csv",index=False)
        y_train.to_csv("data/train/y_train.csv",index=False)
        y_test.to_csv("data/test/y_test.csv",index=False)

        '''
        np.savetxt("data/train/x_train.npy", x_train)
        np.savetxt("data/test/x_test.npy", x_test)
        np.savetxt("data/train/y_train.npy", y_train)
        np.savetxt("data/test/y_test.npy", y_test)
        '''

        return "success"