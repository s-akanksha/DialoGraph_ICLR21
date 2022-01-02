import json
import pdb
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import pickle
from pprint import pprint
import sys, os
import nltk
nltk.download('wordnet')
# Yiheng file for strategies and DA
import inspect
src_file_path = inspect.getfile(lambda: None)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))), 'yiheng_findfeatures/'))
from dialog_acts_extractor import *

train_data = json.load(open('/home/blue_bird/Coding/Dialograph/DialoGraph_ICLR21/data/negotiation_data/train.json', 'r'))
test_data = json.load(open('/home/blue_bird/Coding/Dialograph/DialoGraph_ICLR21/data/negotiation_data/test.json', 'r'))
dev_data = json.load(open('/home/blue_bird/Coding/Dialograph/DialoGraph_ICLR21/data/negotiation_data/dev.json', 'r'))

#diaacts = extract_acts(train_data[0])
new_data = train_data[2085]
for i in range(len(new_data['events'])):
    if type(new_data['events'][i]['data']) == str:
        new_data['events'][i]['data'] = new_data['events'][i]['data'].lower()
#new_data
seqacts = extract_seq_acts(new_data)

my_freq = ddict(int) # id too freq
full_data = train_data + dev_data
for i in range(len(full_data)):
    print (f'{i}/{len(full_data)}', end = '\r')
    for j in range(len(full_data[i]['events'])):
        if type(full_data[i]['events'][j]['data']) == str:
            full_data[i]['events'][j]['data'] = full_data[i]['events'][j]['data'].lower()
    #strats = extract_seq_acts(full_data[i])
    strats = extract_acts(full_data[i])
    strats = strats[1]
    for j in range(len(strats)):
        for idx, val in enumerate(strats[j]):
            my_freq[idx] += val
print ()
print (my_freq)

dialog = train_data[2085]
scene = dialog['scenario']['kbs'][1]['item']['Category'] + ' ' + ' '.join(dialog['scenario']['kbs'][1]['item']['Description']) + ' ' + dialog['scenario']['kbs'][1]['item']['Title']

print(scene)

recommendation_feature_mapping = {"seller_neg_sentiment":0,"seller_pos_sentiment":1,
                                  "buyer_neg_sentiment":2,"buyer_pos_sentiment":3,
                                  "first_person_plural_count_seller":4,"first_person_singular_count_seller":5,
                                  "first_person_plural_count_buyer":6,"first_person_singular_count_buyer":7,
                                  "third_person_singular_seller":8,"third_person_plural_seller":9,
                                  "third_person_singular_buyer":10,"third_person_plural_buyer":11,
                                  "number_of_diff_dic_pos":12,"number_of_diff_dic_neg":13,
                                  "buyer_propose":14,"seller_propose":15,
                                  "hedge_count_seller":16,"hedge_count_buyer":17,
                                  "assertive_count_seller":18,"assertive_count_buyer":19,
                                  "factive_count_seller":20,"factive_count_buyer":21,
                                  "who_propose":22,"seller_trade_in":23,
                                  "personal_concern_seller":24,"sg_concern":25,
                                  "liwc_certainty":26,"liwc_informal":27,
                                  "politeness_seller_please":28,"politeness_seller_gratitude":29,
                                  "politeness_seller_please_s":30,
                                  "ap_des":31,"ap_pata":32,"ap_infer":33,
                                  "family":34,"friend":35,
                                  "politeness_buyer_please":36,"politeness_buyer_gratitude":37,
                                  "politeness_buyer_please_s":38,
                                  "politeness_seller_greet":39,"politeness_buyer_greet":40}

recommendation_feature_reverse_mapping = {v:k for k, v in recommendation_feature_mapping.items()}

def get_strategies(bag_of_strategies):
    '''
    Takes a bag of strategies which is list of (len dialogue) * 41 (features)
    And returns the non one hot encoding
    '''
    return [[recommendation_feature_reverse_mapping[i] for i, val in enumerate(onedia) if val == 1] for onedia in bag_of_strategies]

get_strategies(seqacts[1])

def addStrategies(data):
    '''
    Takes raw data then adds strategies in each event
    '''
    new_data = data.copy()
    for i in range(len(data)):
        #dia_strategies = get_strategies(extract_seq_acts(data[i])[1])
        dia_strategies = get_strategies(extract_acts(data[i])[1])
        # data[i]['events'][j]['strategies'] = []
        for j in range(len(dia_strategies)):
            #if len(new_data[i]['events']) != len(dia_strategies):
            #    pdb.set_trace()
            new_data[i]['events'][j]['strategies'] = dia_strategies[j]
        if i % 100 == 0: print ("Done : {} / {}".format(i, len(data)), end = '\r')
    print ('\n')
    return new_data

data_w_strategies = {
    'train': addStrategies(train_data),
    'test': addStrategies(test_data),
    'dev': addStrategies(dev_data),}
pickle.dump(data_w_strategies, open('/home/blue_bird/Coding/Dialograph/DialoGraph_ICLR21/src/preproc/.ipynb_checkpoints/data_w_strategies.pkl', 'wb'))