import mxnet as mx

class GRU(Block):
    def __init__(self, num_inputs, num_hiddens, batch_first, drop_prob):
        super(Encoder, self).__init__()
        self.drop_prob = drop_prob
        self.batch_first = batch_first
        if batch_first = True:
            self.layout = 'NTC'
        else:
            self.layout = 'TNC'
        self.rnn = rnn.GRU(num_hiddens, layout = self.layout  dropout=drop_prob,bidirectional = True,
          i2h_weight_initializer ='Orthogonal', h2h_weight_initializer= 'Orthogonal'
                           input_size= num_inputs)
                           
     def forward(self,x ,mask):
         #get length of the x
         mask = (x!=0)
         length = mask.sum(1)
