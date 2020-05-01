from datetime import datetime
from logging import DEBUG
from flask import Flask, redirect, url_for, render_template, flash, request, session
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, SelectField, SelectMultipleField, PasswordField
import secrets
from extract_graph_features import GraphFeatures
from preprocessing import Preprocess
from classifiers import Classifier
from sample import Sample
from modelEvaluation import ModelEvaluation
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import threading
import concurrent.futures

gf = GraphFeatures
pre = Preprocess()
classifier = Classifier()
sample = Sample()
me = ModelEvaluation()

app = Flask(__name__)
app.logger.setLevel(DEBUG)
app.config['SECRET_KEY'] = secrets.token_urlsafe(16)


def classification(cid, mid, ml, x_train, x_test, y_train, y_test):
	# Classify here
	clf = {'lr': classifier.logistic_regression, 'dt': classifier.decision_tree_classifier, 'rf': classifier.random_forest, 'svm': classifier.svm, 'xgb': classifier.xg_boost, 'nn':classifier.neural_net}
	# removing customer id before classification; unwanted
	# x_train_ip = x_train.drop(['customer'], axis = 1)
	print("y train shape in classification:", y_train.shape)
	model = clf[ml](x_train, y_train)

	cid_cols = x_test["customer"] == float(cid)
	mid_cols = x_test[mid] == 1

	transaction_idx = x_test[cid_cols & mid_cols].index
	x_test = x_test[cid_cols & mid_cols]
	
	y_test_ip = y_test.loc[transaction_idx]

	#Get the predicted values
	y_pred = model.predict(x_test)

	amt = x_test['amount'].to_numpy()

	amt = amt.reshape((amt.shape[0],1))

	y_pred = y_pred.reshape((y_pred.shape[0],1))
	y_pred[y_pred >= 0.5] = 1
	y_pred[y_pred < 0.5] = 0

	return np.hstack([amt, y_test_ip, y_pred]) 
  

class InputData(Form):

	neo4jPassword = PasswordField('Neo4j DB password', validators=[validators.required()])

	run = SubmitField('Proceed to step 2')
	customerid = TextField('Customer ID', validators=[validators.required()])
	merchantid = TextField('Merchant ID', validators=[validators.required()])
	model = SelectField('Select a model', choices=[('lr', 'Logistic Regression'), ('xgb', 'XG_Boost'), ('rf','Random Forest'), ('nn','Neural Network')])
	results = SubmitField('Get Results')
	graph_results = ""
	original_results = ""

	

	@app.route("/step1", methods = ['GET', 'POST'])
	def step1():
		step1 = InputData(request.form)

		if request.method == 'POST':
			password = request.form['neo4jPassword']
		
			if gf.extractGraphFeatures(password) == 'success':
				# do original dataset preprocessing here
				
				if pre.preprocess_data_ui() == 'success':
					# TODO: show feedback message

					x_test = pd.read_csv('data/test/x_test.csv')  
					y_test = pd.read_csv('data/test/y_test.csv')  
					#x_test_ip = x_test.loc[y_test["fraud"] == 1]
					x_test_ip = x_test
					######## Customers ##########
					customers = x_test_ip[x_test_ip.columns[6]]
					customers = customers.apply(np.int64)

					######## Amount #############
					amount = x_test_ip[x_test_ip.columns[1]]

					######## Merchants ###########
					merchants = x_test_ip[x_test_ip.columns[9:]]
					merchants = merchants.idxmax(axis=1)
					merchants = merchants.replace('merchant_', '', regex=True)

					######## Fraud ###############
					fraud = y_test[y_test.columns[0]]

					result = pd.concat([customers, merchants, amount, fraud], axis=1)
					result.columns = ['CustomerID', 'MerchantID', 'Amount', 'Fraud']

					result.to_csv("data/validation/validation.csv",index=False)
					
					return redirect(url_for('step2'))
			
		return render_template('step1.html', title='Fraud Detection step 1', form = step1)



	@app.route('/step2', methods=['GET', 'POST'])
	def step2():
		step2 = InputData(request.form)

		if request.method == 'POST':
			cid = request.form['customerid']
			mid = request.form['merchantid']
			ml = request.form['model']

			# load all the data
			#graph features
			x_graph_train = pd.read_csv('data/train/x_train.csv')  	
			x_graph_test = pd.read_csv('data/test/x_test.csv')  		

			#non-graph features
			x_original_train = x_graph_train[x_graph_train.columns[6:]]
			x_original_test = x_graph_test[x_graph_test.columns[6:]]

			y_train = pd.read_csv('data/train/y_train.csv')  
			y_test = pd.read_csv('data/test/y_test.csv')  

			# Separate graph and non graph training features

			step2.original_results = "Classification results using original data for transactions between Customer ID " + cid + " and Merchant ID " + mid +":"
			step2.graph_results = "Classification results using graph enhanced data for transactions between Customer ID " + cid + " and Merchant ID " + mid +":"		
		
			mid = "merchant_" + mid	
			
			with concurrent.futures.ThreadPoolExecutor() as executor:
				graphClassification = executor.submit(classification, cid, mid, ml, x_graph_train, x_graph_test, y_train, y_test)
				originalClassification = executor.submit(classification, cid, mid, ml, x_original_train, x_original_test, y_train, y_test)
				graph_res = graphClassification.result()
				original_res = originalClassification.result()
				
			#Display prediction on UI
			InputData.allres = original_res
			InputData.graphres = graph_res
			
		return render_template('step2.html', title='Fraud Detection step 2', form = step2)

	@app.route('/step3', methods=['GET', 'POST'])
	def step3():
		step3 = InputData(request.form)
		if request.method == 'POST':

			# Opening JSON file
			f = open('evaluations/model_evaluation.json')

			# returns JSON object as
			# a dictionary
			data = json.load(f)

			# Iterating through the json
			# list
			InputData.evalMetric = data['evaluation_data'][0]

			# Closing file
			f.close()

		return render_template('step3.html', title='Fraud Detection step 3', form=step3)


if __name__ == '__main__':
    app.run(debug=True)
	