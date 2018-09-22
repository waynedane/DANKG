import pickle
import numpy as np
def getdata(path):
    path =path.split('.')
    if path[-1] =='pkl':
        with open(path,'rb') as f:
            data = pickle.load(f)
    data = np.load(path)
    return data

def process(list_of_string):
    a = ' '.join(list_of_string)
    a= a.replace('\\', '')
    a = a.replace('/', '')
    return a.split(' ')

def padpad(array, max_length):
    length = len(array)
    ndarray = np.pad(array, (0,max_length-length), 'constant', constant_values=0)
    return ndarray

def to_array(instance, dict_):
    content = instace[0]
    keyphrases = instance[1]
    title = content[0]
    abstract = content[1]
    title = process(title)
    abstract = process(abstract)
    title = np.array([dict_[token] for token in title])
    title = padpad(title, 30)
    abstract = np.array([dict_[token] for token in abstract])
    abstract = padpad(abstract,400)
    content = np.concatenate((title,abstract))
    reps = len(keyprhases)
    content = np.tile(content,(reps,1))
    
    for i in range(reps):
        keyphrase = keyphrases[i]
        keyphrase = np.array([dict_[token] for token in keyphrase])
        keyphrase = padpad(keyhprase, 6)
        
    keyphrases = np.array(keyphrases)
    output = np.concatenate((content,kyphrases), -1)
    return output
