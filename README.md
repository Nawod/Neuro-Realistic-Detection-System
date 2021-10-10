# Neuro-Realistic-Detection-System

NLP based malicious traffics and URL classification extension for ELK stack.(elastic search)

***********************************
"nrds_local_run.py" - The deep learning models will run in the local machine for prediction
"nrds_cloud_run.py" - The predictions will get through an API from deep learning model which are deployed on cloud

How to use
1) Clone or download the repo
2) Download and setup the ELK - https://www.elastic.co/start
3) Run the ELK
4) Define the ELK index and Run the nrds_cloud_run.py / nrds_local_run.py
5) Create ELK index pattern for 'nlp_output'
6) Create a dashborad in Kibana to visualize data
