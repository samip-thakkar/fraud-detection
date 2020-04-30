# Fraud Detection using Machine learning and Graph Database

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


