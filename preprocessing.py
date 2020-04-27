# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Import libraries
import pandas as pd
class Preprocess():
    # columns = []
    # from sklearn.preprocessing import MinMaxScaler
    # min_max_scaler = MinMaxScaler()    
    
    """Read the data"""
    def read_data(self):
        df = pd.read_csv('data.csv')
        return df

    """Data Visualization"""
    def data_visualization(self):
        df = self.read_data()
        print(df.nunique())
        target_count = df.fraud.value_counts()
        print('Class 0:', target_count[0])
        print('Class 1:', target_count[1])
        print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
        target_count.plot(kind='bar', title='Count (target)')
        return df
    
    """Data Cleaning"""
    def data_cleaning(self):
        df = self.data_visualization()
        #Drop unnecessary columns
        df = df.drop(['customer', 'zipcodeOri', 'zipMerchant', 'step'], axis = 1)
        
        #Clean the data
        df['age'] = df['age'].apply(lambda x: x[1]).replace('U', 7).astype(int)
        df['gender'] = df['gender'].apply(lambda x: x[1])
        df['merchant'] = df['merchant'].apply(lambda x: x[1:-1])
        df['category'] = df['category'].apply(lambda x: x[1:-1])
        #df['customer'] = df['customer'].apply(lambda x: x[2:-1])
        #Random shuffle
        df = df.sample(frac = 1).reset_index(drop = True)
        return df
    
    """Data Transformation"""
    def data_transformation(self):
        df = self.data_cleaning()
        #Describe the data as features and target class as label
        features = df.drop('fraud', axis = 1)
        label = df.fraud
        features = features[['amount', 'age', 'gender', 'merchant', 'category']]
        features = pd.get_dummies(features, columns = ['age', 'gender', 'merchant', 'category'])
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
    
    """Dimensionality reduction using MCA"""
    def do_mca(self):
        import prince
        x_train, x_test, y_train, y_test = self.scale_data()
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components = 12)
        # x_train = pca.fit_transform(x_train)
        # x_test = pca.transform(x_test)
        return x_train, x_test, y_train, y_test
    
    """Return final preprocessed data"""
    def preprocessed_data(self):
        x_train, x_test, y_train, y_test = self.do_mca() 
        return x_train, x_test, y_train, y_test