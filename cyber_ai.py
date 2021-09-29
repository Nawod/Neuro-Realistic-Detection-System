import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import time

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# create a client instance of the library
elastic_client = Elasticsearch()

#import and train word tokenizer
#import csv from github
# url = "https://raw.githubusercontent.com/Nawod/malicious_url_classifier_api/master/archive/url_train.csv"
# data = pd.read_csv(url)
data = pd.read_csv('archive/url_train.csv')
tokenizer = Tokenizer(num_words=10000, split=' ')
tokenizer.fit_on_texts(data['url'].values)

#load the malicious url classification model
mal_url_model = load_model('mal_url_model.h5')

#retrive elk value
def get_elk_nlp():

    response = elastic_client.search(
        index='test_n',
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


    return nlp_traffic

#text cleaning
def clean_text(df):
    spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]

    for char in spec_chars:
        df['url'] = df['url'].str.replace(char, ' ')

    return df

#tokenize inputs
def token(text):
    X = tokenizer.texts_to_sequences(pd.Series(text).values)
    Y = pad_sequences(X, maxlen=200)
    return Y

#malicious url classification
def url_predict(body):

    #retive the data
    input_data = json.loads(body)['data']

    embeded_text =  token(input_data) #tokenize the data
    predictions = mal_url_model.predict(embeded_text) #classify the data
    
    sentiment = (predictions > 0.5).astype(np.int) #calculate the index of max sentiment
   
    if sentiment==0:
         t_sentiment = 'bad' #set appropriate sentiment
    elif sentiment==1:
         t_sentiment = 'good'

    return { #return the dictionary for endpoint
         "Label" : t_sentiment 
    }

#nlp models prediciton
def nlp_model(df):
    print('Malicious URLs classifing*******')
    
    #text pre processing
    new_df = df
    new_df['url'] = new_df['host'].astype(str).values + new_df['uri'].astype(str).values
    new_df = clean_text(new_df)

    #convert dataframe into a array
    df_array = new_df[['url']].to_numpy()

    # creating a blank series
    label_array = pd.Series([])

    for i in range(df_array.shape[0]):

        #create json requests 
        lists = df_array[i].tolist()
        data = {'data':lists}
        body = str.encode(json.dumps(data))

        #call mal url function to classification
        pred_url = url_predict(body)

        #retrive the outputs
        output = str.encode(json.dumps(pred_url))
        label2 = json.loads(output)['Label']

        #insert labels to series
        label_array[i] = label2
  
    #inserting new column with labels
    df.insert(3, "url_label", label_array)

    return df


#index key values for mal url output
mal_url_keys = [ "ID","host","uri","url_label"]
def nlpFilterKeys(document):
    return {key: document[key] for key in mal_url_keys }

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

    count = 1
    while True:
        print('Batch :', count)
        #retrive data and convert to dataframe
        print('Retrive the data batch from ELK******')
        net_traffic = get_elk_nlp()
        elk_df_nlp = pd.DataFrame(net_traffic)

        #NLP prediction
        nlp_df = nlp_model(elk_df_nlp)

        nlp_df.insert(0, 'ID', range(count , count + len(nlp_df)))

        # Exporting Pandas Data to Elasticsearch
        df_iter = nlp_df.iterrows()
        index, document = next(df_iter)

        helpers.bulk(es_client, nlp_doc_generator(nlp_df))
        print('Batch', count , 'exported to ELK')

        count = count + len(elk_df_nlp)
        # get new records in every 5 seconds
        time.sleep(10)

if __name__ == '__main__':
    main()