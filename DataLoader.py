import os
import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset
import pickle
class mydataset(Dataset):
    def __init__(self,path, set_model):
        self.root = {'train':'train_w2i.npy', 'valida': 'valida_w2i.npy', 'test': 'test_w2i.npy'}
        self.data = np.load(os.path.join(path, self.root[set_model]))
    def __getitem__(self,index):
        title = self.data[index][:20]
        abstract = self.data[index][20:470]
        keyphrase = self.data[index][470:476]
        return title, abstract, keyphrase
    def __len__(self):
        return len(self.data)
