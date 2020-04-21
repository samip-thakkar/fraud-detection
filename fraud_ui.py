# -*- coding: utf-8 -*-
"""

@author: Aishwarya Devulapalli
"""

from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, SelectField, HiddenField, FieldList, FormField
from wtforms.widgets import TextArea
import secrets 
from run_saved import RunSavedModels

DEBUG = True
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_urlsafe(16)

run = RunSavedModels()

class Entries(Form):
	entries = TextField()

#class Results(Form):

#	results = FieldList(FormField(Entries), min_entires=0)	

class InputData(Form):
	customerid = TextField('Customer ID', validators=[validators.required()])
	merchantid = TextField('Merchant ID', validators=[validators.required()])
	model = SelectField('Select a model', choices=[('1', 'Logistic Regression'), ('5','Random Forest'), ('7', 'XgBoost'), ('nn','Neural Network')])
	run = SubmitField('Get Results')
	#results = HiddenField()
	results = TextField() #FieldList(FormField(Entries), min_entires=0)
	res = {}
	allres = []

@app.route("/", methods = ['GET', 'POST'])
def input():
	
	input = InputData(request.form)	
	if request.method == 'POST':
		cid = request.form['customerid']
		mid = request.form['merchantid']
		ml = request.form['model']
		input.results.data = "Classification results for transactions between Customer ID " + cid + " and Merchant ID" + mid +":"		
		mid = "merchant_" + mid	

		InputData.allres = run.run_models(cid, mid, ml)

	return render_template('input.html', title='Fraud Detection', form = input)



if __name__ == "__main__":
	app.run(debug=DEBUG)		