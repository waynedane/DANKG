from mxnet.gluon import nn
import mxnet as mx

class ScaleShift(nn.Block):
    def __init__(self, vocab_size, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.w =self.params.get('weight', shape=(1,self.vocab_size))
        self.b = self.params.get('bias', shape=(1,self.vocab_size),init=mx.init.Zero())
        #self.w = mx.gluon.Parameter('weight', shape=(1, self.vocab_size), init=mx.init.Xavier())
        #self.b = mx.gluon.Parameter('bias', shape=(1, self.vocab_size), init=mx.init.Zero())
    def forward(self,x):
        output =self.w.data()*x+self.b.data()
        return output

class EmbeddingLayer(nn.Block):
    def __init__(self, vocab_size, embedding, **kwargs):
        super(EmbeddingLyaer, self).__init__(**kwargs)
        self.vocab_size = len(embedding)
        self.dim_out = embedding.vec_len
        self.embedding = nn.Embedding(self.vocab_size, self.dim_out)
        self.embedding.initialize()
        self.embedding.weight.set_data = (embedding.idx_to_vec)
    def forward(self,x):
        return self.embedding(x)
        
        
