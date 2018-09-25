from mxnet.gluon import nn
import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.loss import SoftmaxCELos
from mxnet import nd

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
        
from mxnet.gluon.loss import SoftmaxCELoss


class SoftmaxCEMaskedLoss(SoftmaxCELoss):
    """Wrapper of the SoftmaxCELoss that supports valid_length as the input

    """
    def forward(self,  pred, label, valid_length): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        F
        pred : Symbol or NDArray
            Shape (batch_size, length, V)
        label : Symbol or NDArray
            Shape (batch_size, length)
        valid_length : Symbol or NDArray
            Shape (batch_size, )
        Returns
        -------
        loss : Symbol or NDArray
            Shape (batch_size,)
        """
        if self._sparse_label:
            sample_weight = nd.cast(nd.expand_dims(nd.ones_like(label), axis=-1), dtype=np.float32)
        else:
            sample_weight = nd.ones_like(label)
        sample_weight = nd.SequenceMask(sample_weight,
                                       sequence_length=valid_length,
                                       use_sequence_length=True,
                                       axis=1)
        return super(SoftmaxCEMaskedLoss, self).forward( pred, label, sample_weight)
