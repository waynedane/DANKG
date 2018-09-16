import mxnet as mx
from mxnet import nd
from gluonnlp.model import MultiHeadAttentionCell,DotProductAttentionCell
from mxnet.gluon import Block, nn, rnn
import RNN
import random
from customlayer import *
import Constant
base_cell = DotProductAttentionCell(scaled=True, dropout = 0.2)

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
    
    def __init__(self, embedding_dim, head_count, model_dim, drop_prob, dropout):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.head_count = head_count
        self.model_dim = model_dim
        self.drop_prob = drop_prob
        self.dropout = dropout
        self.title_lstm = RNN.LSTM(
            self.embedding_dim, self.model_dim, True, self.drop_prob)
        self.abstract_lstm = RNN.LSTM(
            self.embedding_dim, self.model_dim, True, self.drop_prob)
        self.title_linear = nn.Dense(self.model_dim, flatten = False, in_units= 2*self.model_dim)
        self.abstract_linear = nn.Dense(self.model_dim, flatten = False, in_units= 2*self.model_dim)
        self.final_linear = nn.Dense(2*self.model_dim, flatten = False, in_units= self.model_dim)
        
       
        
        self.ta_mutal = MultiHeadAttentionCell(base_cell=base_cell, 
                                               query_units= 2*self.model_dim, use_bias=True,
                                          key_units = 2*self.model_dim, value_units= 2*self.model_dim, num_heads=self.head_count, weight_initializer= 'Xavier')
        self.at_mutal = MultiHeadAttentionCell(base_cell=base_cell, 
                                               query_units= 2*self.model_dim, use_bias=True,
                                          key_units = 2*self.model_dim, value_units= 2*self.model_dim, num_heads=self.head_count, weight_initializer= 'Xavier')
        self.self_attn = MultiHeadAttentionCell(base_cell=base_cell, 
                                               query_units= 2*self.model_dim, use_bias=True,
                                          key_units = 2*self.model_dim, value_units= 2*self.model_dim, num_heads=self.head_count, weight_initializer= 'Xavier')
        self.ffn1 = Resblock(2*self.model_dim, self.dropout)
        self.ffn2 = Resblock(2*self.model_dim, self.dropout)
        self.W_G = nn.Dense(1, flatten = False, in_units= 4*self.model_dim)
        self.ffn3 = Resblock(2*self.model_dim)
    def forward(self, x, y, x_mask, y_mask):
        l_x = get_length(x_mask)
        l_y = get_length(y_mask)
        h_H, hidden_H= self.title_lstm(x, l_x)
        h_S, hidden_S = self.abstract_lstm(y, l_y)
        x_mask_ = return_mask(x_mask ,y_mask)
        y_mask_ = return_mask(y_mask ,x_mask)
        u_H,_ = self.ta_mutal(h_H, h_S, h_S, x_mask_)
        h_S_hat,_ = self.at_mutal(h_S, h_H, h_H, y_mask_)
        u_H = self.ffn1(u_H)
        G_t = nd.sigmoid(self.W_G(nd.concat(h_S, h_S_hat, dim = -1))).squeeze()
        h_S_ = nd.stack(*[nd.broadcast_mul(i,j.expand_dims(1)) for (i,j) in zip(h_S, G_t)])
        u_S = self.ffn2(h_S_hat + h_S_)
        u_X = nd.concat(u_H, u_S, dim =1)
        mask_u = nd.concat(x_mask,y_mask, dim = -1)
        mask_u = return_mask(mask_u, mask_u)
        u_X, weight = self.self_attn(u_X, u_X, u_X, mask_u)
        u_X = self.ffn3(u_X)
        s = self.final_linear(self.title_linear(hidden_H)+ self.abstract_linear(hidden_S))

        return s, u_X, weight
 
class Decoder(Block):
    def __init__(self, embedding_dim, model_dim, dropout, head_count, vocab_size, extended_size,gpu):
        super(Decoder,self).__init__()
        self.ctx = gpu
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.dropout = dropout
        self.head_count = head_count
        self.vocab_size = vocab_size
        self.extended.size = extended_size
        self.decoder_ltsm = rnn.LSTM(
            2*self.model_dim, layout='NTC', 
            input_size= self.embedding_dim, 
            i2h_weight_initializer= 'Orthogonal',
        h2h_weight_initializer = 'Orthogonal')
        self.self_attn = MultiHeadAttentionCell(base_cell=base_cell, 
                                               query_units= 2*self.model_dim, use_bias=True,
                                          key_units = 2*self.model_dim, value_units= 2*self.model_dim, num_heads=self.head_count, weight_initializer= 'Xavier')
        self.fnn = Resblock(2*self.model_dim, self.dropout)
        self.V1 = nn.Dense(2*self.model_dim, in_units= 3*self.model_dim)
        self.V2 = nn.Dense(self.vocab_size, in_units= 2*self.model_dim)
        self.W_c = nn.Dense(1)
        self.W_s = nn.Dense(1)
        self.W_x = nn.Dense(1)
 

    def forward(self,x, hidden, cell, u_X, indice, mask):
        batch_size = u_X.size(0)
        s_t, (hidden, cell) = self.decoder_lstm(x, (hidden,cell))
        c_t, weight = sel.fnn(self.self_attn(s_t, u_X, u_X, mask))
        weight = weight.squeeze().sum(1)/2
        P_g = nd.softmax(self.V2(self.V1(nd.concat(s_t.squeeze(),c_t.squeeze(),dim=-1))))
        p_g = nd.sigmoid(self.W_c(c_t) + self.W_s(s_t) + self.W_x(x))
        P_g = nd.concat(P_g,nd.zeros(batch_size, self.extended_size),dim = -1)
        p_c = 1-p_g 
        P_g = P_g*p_g.expand_dims(-1)
        weight = weight*p_c.expand_dims(-1)
        P_c = nd.zeros([batch_size, self.extended_size], ctx= self.ctx)
        
        for i in range(batch_size):
            for j in range(self.extend_size):
                P_c[i][indice[i][j]] += weight[i][j]
        P_c = P_c*p_c.expand_dims(-1)
        final_distribution = nd.log_softmax(P_g+P_c)
        
        return final_distribution, hidden, cell, weight, P_g 
    
    def begin_cell(self, hidden)：
        cell = mx.nd.random.uniform(shape = hidden.shape)
        return cell

class seq2seq(block):
    def __init__(self，embedding_dim, head_count, model_dim, drop_prob, dropout, vocab_size, extended_size, gpu, teacher_forcing_ratio=0.5):
        super(seq2seq, self).__init__()
        self.ctx = gpu
        self.embedding_dim = embedding_dim
        self.head_count = head_count
        self.model_dim = model_dim
        self.drop_prob =drop_prob
        self.dropout = dropout
        self.vocab_size = vcab_size
        self.extended_size = extended_size
        self.teacher_forcing = teacher_forcing_ratio
        self.embedding = EmbeddingLayer
        self.encoder = Encoder(self.embedding_dim, self.head_count, self.model_dim, self.drop_prob, self.dropout)
        self.decoder = Decoder(self.embedding_dim, self.model_dim, self.dropout, self.head_count, self.vocab_size, self.extended_size,self.ctx)
        self.loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss(from_logits = True)
    def forward(self,x_ti, x_ab, ti_mask, ab_mask, trg, indice):
        cur_batch_size = ti_mask.shape[0]
        ti_input, ab_input = embedding(x_ti), embedding(x_ab)
        decoder_state, encoder_outputs, _ = encoder(ti_input, ab_input, ti_mask, ab_mask)
        mask = nd.concat(ti_mask, ab_mask,dim=-1)
        mask = return_mask(mask, nd.ones(cur_batch_size,1))
        cell = decoder.begin_cell()
        decoder_input = embedding(nd.array([Constant.bos]*cur_batch_size))
        P_g_list =[]
        loss_total = 0
        for i in range(len(trg)):
            prediction, decoder_state, cell, weight,P_g= decoder(decoder_input, decoder_state, cell, encoder_outputs, indice, mask)
            P_g_list.append(P_g)
            loss_mask = (trg[i]！=2)
            is_teacher = random.random() < self.teacher_forcing
            decoder_input = embedding(trg[i]) if is_teacher else embedding(prediction.argmax(axis=1))
            loss =self.loss(prediction, trg[i])*loss_mask
            loss =loss.sum()
            loss_total = loss_total+loss
        loss_total = loss_total/len(y)
    
    
def seq2seq(embedding, encoder, decoder, x_ti, x_ab ,ti_mask ,ab_mask, trg, indcie):
    loss_fun = mxnet.gluon.loss.SoftmaxCrossEntropyLoss(from_logits = True)
    cur_batch_size = ti_mask.shape[0]
    ti_input, ab_input = embedding(x_ti), embedding(x_ab)
    decoder_state, encoder_outputs, _ = encoder(ti_input, ab_input, ti_mask, ab_mask)
    mask = nd.concat(ti_mask, ab_mask,dim=-1)
    mask = return_mask(mask, nd.ones(cur_batch_size,1))
    cell = decoder.begin_cell()
    decoder_input = embedding(nd.array([Constant.bos]*cur_batch_size))
    P_g_list =[]
    loss_total = 0
    for i in range(len(trg)):
        prediction, decoder_state, cell, weight,P_g= decoder(decoder_input, decoder_state, cell, encoder_outputs, indice, mask)
        P_g_list.append(P_g.sum(0)/cur_batch_size)
        loss_mask = (trg[i]！=0)
        is_teacher = random.random() < self.teacher_forcing
        decoder_input = embedding(trg[i]) if is_teacher else embedding(prediction.argmax(axis=1))
        loss = loss_fun(prediction, trg[i])*loss_mask
        loss =loss.sum()
        loss_total = loss_total+loss
    loss_total = loss_total/len(y)
    return loss_total, P_g_list
        
    
    
            
           
        
