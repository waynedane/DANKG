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
                           
    def forward(self, x, hidden = None, length):
         #get length of the x
        
         #feed forward
        outputs, _ = self.rnn(x) #outputs:[batch, seq_length, 2*num_hiddens]
        if hidden != None:
            outputs, _ = self.rnn(x,hidden)
        outputs = outputs.transpose(1,0,2)
        outputs = nd.SequenceMask(outputs, sequence_length=lenghth,use_sequence_length=True, value=0)
        hidden = nd.stack([outputs[:,i,:] for i in length])
        
        return outputs.transpose(1,0,2), hidden
    
    
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
        
   def forward(self, x,length):
      
        outputs, _ = self.rnn(x) #outputs:[batch, seq_length, 2*num_hiddens]
        outputs = outputs.transpose(1,0,2)
        outputs = nd.SequenceMask(outputs, sequence_length=lenghth,use_sequence_length=True, value=0)
        hidden = nd.stack([outputs[:,i,:] for i in length])
        
        return outputs.transpose(1,0,2), hidden
