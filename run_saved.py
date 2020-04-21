# -*- coding: utf-8 -*-
"""

@author: Aishwarya Devulapalli
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from keras.models import load_model
import pickle
from classifiers import Classifier

classifier = Classifier()

class RunSavedModels:

	def save_models(self, model, name):
	    
	    print("Saving the model")
	    filename = "models/" + name
	    pickle.dump(model, open(filename, 'wb'))        

	def load_models(self, name):

	    print("Loading the model")
	    filename = "models/" + name
	    if(name == "nn.h5"):
	        return load_model(filename)
	        
	    return pickle.load(open(filename, 'rb'))

	def run_models(self, cid, mid, choice):

		if(choice == 1):
			name = "lr.sav"
		elif(choice == 5):
			name = "rf.sav"
		elif(choice == 7):
			name = "xg.sav"
		else:
			name = "nn.h5"

		model = self.load_models(name)	

		#Get all transactions for the input customer ID & merchant ID
		test_data = pd.read_csv("data/test_data.csv", index_col=0)
		x_test = test_data.loc[:, :'category_es_wellnessandbeauty']
		y_test = test_data.loc[:, 'fraud':]

		cid_cols = x_test["customer"] == float(cid)
		mid_cols = x_test[mid] == 1

		transaction_idx = x_test[cid_cols & mid_cols].index
		x_test = x_test[cid_cols & mid_cols]
		min_max = MinMaxScaler()
		x_test_ip = min_max.fit_transform(x_test)

		y_test_ip = y_test.loc[transaction_idx]

		y_pred = model.predict(x_test_ip)
		amt = x_test['amount'].to_numpy()
		amt = amt.reshape((amt.shape[0],1))
		y_pred[y_pred >= 0.5] = 1
		y_pred[y_pred < 0.5] = 0

		return np.hstack([amt, y_test_ip, y_pred]) 