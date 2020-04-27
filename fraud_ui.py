from datetime import datetime
from logging import DEBUG
from flask import Flask, redirect, url_for, render_template, flash, request, session
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, SelectField, SelectMultipleField
import secrets
from extract_graph_features import GraphFeatures
from preprocessing import Preprocess
from classifiers import Classifier
from sample import Sample
from modelEvaluation import ModelEvaluation
import pandas as pd 

gf = GraphFeatures
pre = Preprocess()
classifier = Classifier()
sample = Sample()
me = ModelEvaluation()

app = Flask(__name__)
app.logger.setLevel(DEBUG)


app.config['SECRET_KEY'] = secrets.token_urlsafe(16)

class InputData(Form):

	neo4jPassword = TextField('Neo4j DB password', validators=[validators.required()])

	run = SubmitField('Proceed to step 2')
	customerid = TextField('Customer ID', validators=[validators.required()])
	merchantid = TextField('Merchant ID', validators=[validators.required()])
	model = SelectField('Select a model', choices=[('lr', 'Logistic Regression'), ('svm','SVM'), ('xgb', 'XG_Boost'), ('rf','Random Forest'), ('nn','Neural Network')])
	results = SubmitField('Get Results')


	@app.route("/step1", methods = ['GET', 'POST'])
	def step1():
		step1 = InputData(request.form)

		if request.method == 'POST':
			password = request.form['neo4jPassword']
		
			if gf.extractGraphFeatures(password) == 'success':
				# do original dataset preprocessing here
				if pre.preprocess_data_ui() == 'success':
					# TODO: show feedback message
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
			x_train = pd.read_csv('data/train/x_train.csv')  
			x_test = pd.read_csv('data/test/x_test.csv')  
			y_train = pd.read_csv('data/train/y_train.csv')  
			y_test = pd.read_csv('data/test/y_test.csv')  

			# Classify here
			clf = {'lr': classifier.logistic_regression, 'dt': classifier.decision_tree_classifier, 'rf': classifier.random_forest, 'svm': classifier.svm, 'xgb': classifier.xg_boost, 'nn':classifier.neural_net}
			model = clf[ml](x_train, y_train)

			#Get the predicted values
			y_pred = model.predict(x_test)

			#Get the model evaluation
			me.modelevaluation(y_test.to_numpy(), y_pred)

		return render_template('step2.html', title='Fraud Detection step 2', form = step2)


if __name__ == '__main__':
    app.run(debug=True)