import pickle
import numpy as np
def getdata(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data
def process(list_of_string):
    a = ' '.join(list_of_string)
    a= a.replace('\\', '')
    a = a.replace('/', '')
    return a.split(' ')
def to_array(instance, dict_):
    content = instace[0]
    keyphrases = instance[1]
    title = content[0]
    abstract = content[1]
    title = process(title)
    abstract = process(abstract)
    title = np.array([dict_[token] for token in title])
    abstract = np.array([dict_[token] for token in abstract])
    
