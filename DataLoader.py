import os
import mxnet as mx
from mxnet.gluon.data import Dataset
import pickle
class mydataset(Dataset):
    def __init__(self,path, set_model):
        self.root = {'train':'trainset', 'valida': 'validaset', 'test': 'testset'}
        self.path = os.path.join(path, self.root[set_model])
        with open(self.path,'rb') as f:
            data = pickle.load(f)
        self.data = data
    def __getitem__(self,index):
        title = self.data[index][:20]
        abstract = self.data[index][20:470]
        keyphrase = self.data[index][470:476]
        return title, abstract, keyphrase
    def __len__(self):
        return len(self.data)
