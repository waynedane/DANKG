import mxnet as mx
import model
import Constant
import myDataLoader
import utils.py
from mxnet.gluon.data import DataLoader

#设置超参数

Embedding_Dim, Vocab_Size, Extended_Size, Model_Dim, Head, drop_prob, dropout, tearcher_forcing, learning_rate =\
Constant.Embedding_Dim, Constant.Vocab_Size, Constant.Extended_Size,\ 
Constant.Model_Dim, Constant.Head, Constant.drop_prob,\
Constant.dropout, Constant.tearcher_forcing, Constant.learning_rate
#定义gpu
ctx = mxnet.gpu()

#初始化模型
net = model.seq2seq(Embedding_Dim, Head, Model_Dim, drop_prob, dropout, Vocab_Size, Extended_Size, tearcher_forcing)
net.initialize(ctx=mx.gpu())

#创建数据迭代器
dataset = myDataLodaer('/home/dwy/DKGMA_data', 'train')

for index, instance in enumerate(dataset):
    t,a,k = instance.as_in_context(ctx)
    t_indice, a_indice = utils.bucket(t), utils.bucket(a)
    title, abstract = utils.unk(t_indice), utils.unk(a_indice)
    t_mask, a_mask = (t_indice!=0), (a_indice!=0)
    decoder_mask = mx.nd.concat(t_mask, a_mask, -1)
    
    
