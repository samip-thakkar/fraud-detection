from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, SelectField
import secrets
import ML_models 


DEBUG = True
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_urlsafe(16)


class InputData(Form):
	customerid = TextField('Customer ID', validators=[validators.required()])
	merchantid = TextField('Merchant ID', validators=[validators.required()])
	model = SelectField('Select a model', choices=[('lr', 'Logistic Regression'), ('svm','SVM'), ('rf','Random Forest'), ('nn','Neural Network')])
	run = SubmitField('Get Results')

	@app.route("/", methods = ['GET', 'POST'])
	def input():
		input = InputData(request.form)


		if request.method == 'POST':
			cid = request.form['customerid']
			mid = request.form['merchantid']
			ml = request.form['model']
			ML_models.run_model(cid, mid, ml)
			print(cid, mid, ml)

		return render_template('input.html', title='Fraud Detection', form = input)



if __name__ == "__main__":
	app.run()		