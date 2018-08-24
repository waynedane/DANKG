import mxnet as mx
import gluonnlp.model.MultiHeadAttentionCell as multihead
from mxnet.gluon import Block, nn

import mxnet as mx
from mxnet import nd
from gluonnlp.model import MultiHeadAttentionCell,DotProductAttentionCell
from mxnet.gluon import Block, nn, rnn

base_cell = DotProductAttentionCell(scaled=True, dropout = 0.5)

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

class Encoder(Block):
    
    def __init__(self, embedding_dim, head_count, model_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.head_count = head_count
        self.model_dim = model_dim
        self.title_lstm = rnn.LSTM(
            self.model_dim, layout='NTC', 
            bidirectional=True, input_size= self.embedding_dim, 
            i2h_weight_initializer= 'Orthogonal',
        h2h_weight_initializer = 'Orthogonal')
        self.abstract_lstm = rnn.LSTM(
            self.model_dim, layout='NTC', 
            bidirectional=True, input_size= self.embedding_dim, 
            i2h_weight_initializer= 'Orthogonal',
        h2h_weight_initializer = 'Orthogonal')
        self.title_linear = nn.Dense(self.model_dim, in_units= 2*self.model_dim)
        self.abstract_linear = nn.Dense(self.model_dim, in_units= 2*self.model_dim)
        self.final_linear = nn.Dense(self.model_dim, in_units= self.model_dim)
        self.ta_mutal = MultiHeadAttentionCell(base_cell=base_cell, 
                                               query_units= self.model_dim, use_bias=True,
                                          key_units = self.model_dim, value_units= self.model_dim, num_heads=8)
        self.ta_mutal = MultiHeadAttentionCell(base_cell=base_cell, 
                                               query_units= self.model_dim, use_bias=True,
                                          key_units = self.model_dim, value_units= self.model_dim, num_heads=8)
        self.ffn1 = ResBlock(2*self.model_dim)
        self.ffn2 = ResBlock(2*self.model_dim)
        self.W_G = nn.Dense(1, in_units= 4*self.model_dim)
        self. ffn3 = ResBlock(2*self.model_dim)
    def forward(self, x, y, x_mask, y_mask):
        h_H,_ = self.title_lstm(x)
        h_S,_ = self.abstract_lstm(y)
        u_H,_ = self.ta_mutal(h_S, h_S, h_H, y_mask)
        h_S_hat,_ = self.at_mutal(h_H, h_H, h_S, x_mask)
        u_H = self.ffn1(u_H)
        return h_H
