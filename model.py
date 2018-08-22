import mxnet as mx
import gluonnlp.model.MultiHeadAttentionCell as multihead
from mxnet.gluon import Block, nn

class Resblock(Block):
    
    def __init__(self, model_dim, dropout =0.1):
        super(Resblock, self).__init__()
        self.model_dim = model_dim
        self.dropout = dropout
        self.resblock = nn.Sequential()
        with self.resblock.name_scope():
            self.resblock.add(nn.LayerNorm())
            self.resblock.add(nn.Dense(2*self.model_dim,in_units= self.model_dim,activation="relu"))
            self.resblock.add(nn.Dropout(self.dropout))
            self.resblock.add(nn.Dense(self.model_dim,in_units = 2*self.model_dim))
            self.resblock.add(nn.Dropout(self.dropout))
            
    def forward(self, x):
        output = self.resblock(x)
        return output+x
