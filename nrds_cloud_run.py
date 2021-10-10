from typing import Dict, List
from google.cloud import aiplatform
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# create a client instance of the library
elastic_client = Elasticsearch()

print('************************************')
#import and train word tokenizer
#import csv from github
# url = "https://raw.githubusercontent.com/Nawod/malicious_url_classifier_api/master/archive/url_train.csv"
# url_data = pd.read_csv(url)
#url dataset
url_data = pd.read_csv('archive/url_combined_train.csv')
url_tokenizer = Tokenizer(num_words=50000, split=' ')
url_tokenizer.fit_on_texts(url_data['url'].values)

#traffic dataset
traffic_data = pd.read_csv('archive/traffic_combine_train.csv')
traffic_tokenizer = Tokenizer(num_words=5000, split=' ')
traffic_tokenizer.fit_on_texts(traffic_data['feature'].values)
print('Tokenizer Loaded')
print('************************************')

#retrive elk value
def get_elk_nlp():

    response = elastic_client.search(
        index='nlp-log-1', #elasticsearch index
        body={},
    )
    # print(type(response))

    # nested inside the API response object
    elastic_docs_nlp = response["hits"]["hits"]

    traffics_n = elastic_docs_nlp
    nlp_traffic = {}
    #append data
    for num, doc in enumerate(traffics_n):
        traffic = doc["_source"]

        for key, value in traffic.items():
            if key == "@timestamp":
                try:
                    nlp_traffic[key] = np.append(nlp_traffic[key], value)
                except KeyError:
                    nlp_traffic[key] = np.array([value])
            if key == "method":
                try:
                    nlp_traffic[key] = np.append(nlp_traffic[key], value)
                except KeyError:
                    nlp_traffic[key] = np.array([value])
            if key == "id_resp_p":
                try:
                    nlp_traffic[key] = np.append(nlp_traffic[key], value)
                except KeyError:
                    nlp_traffic[key] = np.array([value])
            if key == "version":
                try:
                    nlp_traffic[key] = np.append(nlp_traffic[key], value)
                except KeyError:
                    nlp_traffic[key] = np.array([value])
            if key == "host":
                try:
                    nlp_traffic[key] = np.append(nlp_traffic[key], value)
                except KeyError:
                    nlp_traffic[key] = np.array([value])
            if key == "uri":
                try:
                    nlp_traffic[key] = np.append(nlp_traffic[key], value)
                except KeyError:
                    nlp_traffic[key] = np.array([value])
            if key == "user_agent":
                try:
                    nlp_traffic[key] = np.append(nlp_traffic[key], value)
                except KeyError:
                    nlp_traffic[key] = np.array([value])
            if key == "status_msg":
                try:
                    nlp_traffic[key] = np.append(nlp_traffic[key], value)
                except KeyError:
                    nlp_traffic[key] = np.array([value])
            if key == "response_body_len":
                try:
                    nlp_traffic[key] = np.append(nlp_traffic[key], value)
                except KeyError:
                    nlp_traffic[key] = np.array([value])


    return nlp_traffic

#text cleaning
def url_clean_text(df):
    spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              ">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]

    for char in spec_chars:
        df['url'] = df['url'].str.replace(char, ' ')

    return df

def traffic_clean_text(df):
    spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              ">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]

    for char in spec_chars:
        df['feature'] = df['feature'].str.replace(char, ' ')

    return df

#tokenize inputs
def url_token(text):
    X = url_tokenizer.texts_to_sequences(pd.Series(text).values)
    Y = pad_sequences(X, maxlen=150)
    return Y

def traffic_token(text):
    X = traffic_tokenizer.texts_to_sequences(pd.Series(text).values)
    Y = pad_sequences(X, maxlen=160)
    return Y

#vertex AI API call for get prediction
def predict_mal_url_api(

    project: str ="511747508882", 
    endpoint_id: str ="1302648600030871552",
    location: str = "us-central1",
    instances: List = list
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint_id)
    prediction = endpoint.predict(instances=instances)
    
    return prediction

def predict_net_traffic_api(

    project: str ="511747508882", 
    endpoint_id: str ="3266218037564407808",
    location: str = "us-central1",
    instances: List = list
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint(endpoint_id)
    prediction = endpoint.predict(instances=instances)
    
    return prediction

#malicious url classification
def url_predict(body):

    #retive the data
    input_data = json.loads(body)['data']

    embeded_text =  url_token(input_data) #tokenize the data
    list = embeded_text.tolist()
    response = predict_mal_url_api(instances=list) #call vertex AI API
    prediction = response.predictions[0][0] #retrive data

    # print('url prediction: ', prediction)

    #set appropriate sentiment
    if prediction < 0.5:
         t_sentiment = 'Malicious URL' 
    elif prediction >= 0.5:
         t_sentiment = 'good'

    return { #return the dictionary for endpoint
         "Label" : t_sentiment 
    }

#network traffic classification
def traffic_predict(body):

    #retive the data
    input_data = json.loads(body)['data']

    embeded_text =  traffic_token(input_data) #tokenize the data
    list = embeded_text.tolist()
    response = predict_net_traffic_api(instances=list) #call vertex AI API
    prediction = response.predictions[0][0] #retrive data

    # print('traffic prediction: ', prediction)

    #set appropriate sentiment
    if prediction < 0.5:
         t_sentiment = 'bad traffic' 
    elif prediction >= 0.5:
         t_sentiment = 'good'

    return { #return the dictionary for endpoint
         "Label" : t_sentiment 
    }

#nlp models prediciton
def nlp_model(df):
    print('Network traffic classifing_#####')
    
    #text pre processing
    new_df = df
    new_df['url'] = new_df['host'].astype(str).values + new_df['uri'].astype(str).values
    new_df['feature'] = new_df['method'].astype(str).values+' '+new_df['id_resp_p'].astype(str).values+' '+new_df['version'].astype(str).values+' '+new_df['host'].astype(str).values+' '+new_df['uri'].astype(str).values+' '+new_df['user_agent'].astype(str).values+' '+new_df['status_msg'].astype(str).values+' '+new_df['response_body_len'].astype(str).values
    new_df = url_clean_text(new_df)
    new_df= traffic_clean_text(new_df)
    
    #convert dataframe into a array
    url_array = new_df[['url']].to_numpy()
    traffic_array = new_df[['feature']].to_numpy()

    # creating a blank series
    url_label_array = pd.Series([])
    traffic_label_array = pd.Series([])

    for i in range(url_array.shape[0]):
        #create json requests 
        url_lists = url_array[i].tolist() #for urls
        url_data = {'data':url_lists}
        url_body = str.encode(json.dumps(url_data))

        traffic_lists = traffic_array[i].tolist() #for net traffics
        traffic_data = {'data':traffic_lists} 
        traffic_body = str.encode(json.dumps(traffic_data))

        #call mal url function to classification
        pred_url = url_predict(url_body)
        pred_traffic = traffic_predict(traffic_body)

        #retrive the outputs
        url_output = str.encode(json.dumps(pred_url))
        url_label = json.loads(url_output)['Label']

        traffic_output = str.encode(json.dumps(pred_traffic))
        traffic_label = json.loads(traffic_output)['Label']

        #insert labels to series
        url_label_array[i] = url_label
        traffic_label_array[i] = traffic_label
  
    #inserting new column with labels
    df.insert(1, "url_label", url_label_array)
    df.insert(2, "traffic_label", traffic_label_array)

    return df

#index key values for mal url output
nlp_keys = [ "@timestamp","ID","method","id_resp_p","version","host","uri","user_agent","status_msg","response_body_len","url_label","traffic_label"]
def nlpFilterKeys(document):
    return {key: document[key] for key in nlp_keys }

# es_client = Elasticsearch(http_compress=True)
es_client = Elasticsearch([{'host': 'localhost', 'port': 9200}])

def nlp_doc_generator(df):
    df_iter = df.iterrows()
    for index, document in df_iter:
        yield {
                "_index": 'nlp_output',
                "_type": "_doc",
                "_id" : f"{document['ID']}",
                "_source": nlpFilterKeys(document),
            }
    #raise StopIteration


#main loop
def main():
    # u = url_data.sample(n = 10).reset_index(drop=True)
    # t = traffic_data.sample(n = 10).reset_index(drop=True)
    # print(u.head())
    # print(t.head())
    count = 1
    while True:
        print('Batch :', count)
        # start_time = time.time() #to calculate response time
        #retrive data and convert to dataframe
        print('Retriving the data batch from ELK_#####')

        nlp_traffic = get_elk_nlp()
        elk_df_nlp = pd.DataFrame(nlp_traffic)
        # elk_df_nlp = u
        # elk_df_nlp['feature'] = pd.Series(t['feature'])
        
        # print(elk_df_nlp.head())
        #NLP prediction
        nlp_df = nlp_model(elk_df_nlp)

        nlp_df.insert(0, 'ID', range(count , count + len(nlp_df)))
        # print(nlp_df.head())
        # print(nlp_df.shape)

        # Exporting Pandas Data to Elasticsearch
        helpers.bulk(es_client, nlp_doc_generator(nlp_df))

        # print("response time %s seconds ---" % (time.time() - start_time))
        print('Batch', count , 'exported to ELK')
        print('************************************')

        count = count + len(elk_df_nlp)
        # get new records in every 10 seconds
        time.sleep(10)

if __name__ == '__main__':
    main()