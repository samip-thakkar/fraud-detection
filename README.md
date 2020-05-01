# Fraud Detection using Machine learning and Graph Databases

## Why graph databases?

Graph databases are indexed naturally by relationships which makes it easier for users to include data without having to do much modeling in advance. This makes graph technology very useful for keeping up with the deception and speed of fraudsters.
Traditional fraud prevention measures focus on discrete data points such as specific accounts, individuals, devices or IP addresses. However, today’s sophisticated fraudsters escape detection by forming fraud rings comprised of stolen and synthetic identities. To uncover such fraud rings, it is essential to look beyond individual data points to the connections that link them.

## Features used for classfication

~~~
[x] Amount
[x] CustomerID 
[x] MerchantID
[x] Category of transaction
~~~

## Infrastructure setup

We provisioned a single GCP VM instance for hosting Neo4j instance. 

My configuration is as follows: 


| Node        | OS              | Internal IP    | Disk                     | RAM    | External IP   |
| ------------|:---------------:|:--------------:|:------------------------:|:------:|:-------------:| 
| neo4j       | Ubuntu 16.04TLS | 10.168.0.10    | SSD                      | 7.5GB  | 35.236.81.95  |

Our client side application is implemented in Flask framework which can interact with this GCP Neo4j instance via http and https ports.

## Execution

1) Please ensure you have Python 3.7 installed.
2) Clone this repository in the local environment.
3) You can download all the requisite libraries by executing the command : `pip3 install -r requirements.txt`
4) You can run the application by then running : `python3 fraud_ui.py` which is the starting point.
5) On the first screen you'll be asked to enter password which will help you connect to the remote Neo4j instance for graph features extraction. Please enter `semantic` as the password.
6) On the second screen you'll have to input a merchant ID and a customer ID for classification.
7) For this you need to go the directory inside our application : `data/validation/validation.csv` and choose any random  transaction pair with the customer and merchant ID.
8) Please enter the values in the respective input boxes and click submit.
9) After some time you should be able to see the classification results being displayed on the screen itself. You need to wait for a couple of minutes for the model to finish its runtime classification. (Random forest is the fastest one)
10) The prediction results should be now displayed with the actual values and the predicted ones.
11) User can now navigate to the screen 3 where they can compare the evaluation metrics like accuracy, recall and precision for fraud transactions based on both original dataset and the dataset augmented with graph features OR the user can find the model evaluation metrics in the file `model_evaluations.json` under `evaluations` folder.

