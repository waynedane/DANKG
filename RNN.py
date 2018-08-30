import mxnet as mx
from mxnet import nd
import mxnet.gluon.rnn as rnn
class GRU(Block):
    def __init__(self, num_inputs, num_hiddens, batch_first, drop_prob):
        super(GRU, self).__init__()
        self.drop_prob = drop_prob
        self.batch_first = batch_first
        if batch_first = True:
            self.layout = 'NTC'
        else:
            self.layout = 'TNC'
        self.rnn = rnn.GRU(num_hiddens, layout = self.layout  dropout=drop_prob,bidirectional = True,
          input_size= num_inputs, i2h_weight_initializer ='Orthogonal', h2h_weight_initializer= 'Orthogonal'
                        )
                           
    def forward(self, x, mask):
         #get length of the x
        mask = (x!=0)
        mask = mask.expand_dims(axis=-1)
        length = mask.sum(1)
         #feed forward
        outputs, _ = self.rnn(x) #outputs:[batch, seq_length, 2*num_hiddens]
        outputs = outputs*mask
        hidden = nd.stack([outputs[:,i,:] for i in length])
        
        return outputs, hidden
    
    
class LSTM(Block):
    def __init__(self, num_inputs, num_hiddens, batch_first, drop_prob):
        super(LSTM, self).__init__()
        self.drop_prob = drop_prob
        self.batch_first = batch_first
        if batch_first = True:
            self.layout = 'NTC'
        else:
            self.layout = 'TNC'
         self.rnn = rnn.LSTM(num_hiddens, layout = self.layout  dropout=drop_prob,bidirectional = True,
                 input_size= num_inputs, i2h_weight_initializer ='Orthogonal', h2h_weight_initializer= 'Orthogonal'
                           )
        
   def forward(self, x, mask):
        mask = (x!=0)
        mask = mask.expand_dims(axis=-1)
        length = mask.sum(1)
        outputs, _ = self.rnn(x) #outputs:[batch, seq_length, 2*num_hiddens]
        outputs = outputs*mask
        hidden = nd.stack([outputs[:,i,:] for i in length])
        
        return outputs, hidden
