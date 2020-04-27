from datetime import datetime
from logging import DEBUG
from flask import Flask, redirect, url_for, render_template, flash, request, session
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, SelectField, SelectMultipleField
import secrets
import ML_models_saved 
from extract_graph_features import GraphFeatures
from preprocessing import Preprocess

gf = GraphFeatures
pre = Preprocess()

app = Flask(__name__)
app.logger.setLevel(DEBUG)


app.config['SECRET_KEY'] = secrets.token_urlsafe(16)

class InputData(Form):

	neo4jPassword = TextField('Neo4j DB password', validators=[validators.required()])

	run = SubmitField('Proceed to step 2')
	customerid = TextField('Customer ID', validators=[validators.required()])
	merchantid = TextField('Merchant ID', validators=[validators.required()])
	model = SelectField('Select a model', choices=[('lr', 'Logistic Regression'), ('svm','SVM'), ('rf','Random Forest'), ('nn','Neural Network')])
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
		#return session['graphFeatures']
		step2 = InputData(request.form)
		if request.method == 'POST':
			cid = request.form['customerid']
			mid = request.form['merchantid']
			ml = request.form['model']
			ML_models_saved.run_model(cid, mid, ml)
			print(cid, mid, ml)

		return render_template('step2.html', title='Fraud Detection step 2', form = step2)


if __name__ == '__main__':
    app.run(debug=True)