import mxnet as mx
import model
import Constant
import myDataLoader
import utils.py
from mxnet.gluon.data import DataLoader
Embedding_Dim, Vocab_Size, Extended_Size, Model_Dim, Head, drop_prob, dropout, tearcher_forcing, learning_rate =\
Constant.Embedding_Dim, Constant.Vocab_Size, Constant.Extended_Size,\ 
Constant.Model_Dim, Constant.Head, Constant.drop_prob,\
Constant.dropout, Constant.tearcher_forcing, Constant.learning_rate
#定义gpu
ctx = mxnet.gpu()
#初始化模型
net = model.seq2seq(Embedding_Dim, Head, Model_Dim, drop_prob, dropout, Vocab_Size, Extended_Size, tearcher_forcing)
net.initialize(ctx=mx.gpu())


dataset = myDataLodaer('/home/dwy/DKGMA_data', 'train')
for index, instance in enumerate(dataset):
    t,a,k = instance.as_in_context(ctx)
    title = utils.get_batch(t)
    abstract = utils.get_batch(a)
    
