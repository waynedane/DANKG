from mxnet.gluon import nn

class ScaleShift(nn.Block):
    def __init__(self, vocab_size):
        super(ScaleShift, self).__init__()
        self.vocab_size = vocab_size
        self.w =self.params.get('weight', shape=(1,self.vocab_size))
        self.b = self.params.get('bias', shape=(1,self.vocab_size),init=mx.init.Zero())
        #self.w = mx.gluon.Parameter('weight', shape=(1, self.vocab_size), init=mx.init.Xavier())
        #self.b = mx.gluon.Parameter('bias', shape=(1, self.vocab_size), init=mx.init.Zero())
    def forward(self,x):
        out =self.w.data(x.context)*x+self.b.data(x.context)
        return out
