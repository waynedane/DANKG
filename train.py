import mxnet as mx
import model
import Constant
import myDataLoader
import utils.py
from mxnet.gluon.data import DataLoader

#定义验证函数
def validation(dataset):
    total_loss = 0
    data = DataLoader(dataset, batch_size=256, shuffle=False, sampler=None, 
                      last_batch='keep')
        for index, instance in enumerate(data):
            batchsize = instance.shape(0)
            t,a,k = instance.as_in_context(ctx)
            t_indice, a_indice = utils.bucket(t), utils.bucket(a)
            title, abstract = utils.unk(t_indice), utils.unk(a_indice)
            t_mask, a_mask = (t_indice!=0), (a_indice!=0)
            indice = mx.nd.concat(t_indice, a_indice, -1)
            loss = net(title, abstract, t_mask, a_mask, indice)
            total_loss += loss.mean().asscalar()
    valida_loss = totola_loss/dataset.len_
    return valida_loss
    

#设置超参数

Embedding_Dim, Vocab_Size, Extended_Size, Model_Dim, Head, drop_prob, dropout, tearcher_forcing, learning_rate =\
Constant.Embedding_Dim, Constant.Vocab_Size, Constant.Extended_Size,\ 
Constant.Model_Dim, Constant.Head, Constant.drop_prob,\
Constant.dropout, Constant.tearcher_forcing, Constant.learning_rate
#定义gpu
ctx_0 = mxnet.gpu()

#初始化模型
#net = model.seq2seq(Embedding_Dim, Head, Model_Dim, drop_prob, dropout, Vocab_Size, Extended_Size, ctx_0, tearcher_forcing)
embedding = 
encoder =
decoder =
net.initialize(ctx= ctx_0)

#创建数据迭代器
trainset = myDataLodaer('/home/dwy/DKGMA_data', 'train')
validaset = myDataLodaer('/home/dwy/DKGMA_data', 'valida')
# 优化器
embedding.embedding.collect_params().setattr('grad_req', 'null') #预训练的词嵌入层不参与训练。
en_trainer = gluon.Trainer(encoder.collect_params(), 'adam', {'lr': 1e-5, 'grad_clip': 2})
de_trainer = gluon.Trainer(decoder.collect_params(), 'adam', {'lr': 1e-5, 'grad_clip': 2})
from time import time
tic = time()
total_loss = .0
epoch = 0
traindata = DataLoader(trainset, batch_size=64, shuffle=True, sampler=None, 
                      last_batch='keep', num_workers = 2)
for index, instance in enumerate(traindata):
    
    batchsize = instance.shape(0)
    t,a,k = instance.as_in_context(ctx)
    t_indice, a_indice = utils.bucket(t), utils.bucket(a)
    #title, abstract = utils.unk(t_indice), utils.unk(a_indice)
    t_mask, a_mask = (t_indice!=0), (a_indice!=0)
    indice = mx.nd.concat(t_indice, a_indice, -1)
    
    
    with mx.autograd.record:
        title, abstract = embedding(title), embedding(abstract)
        loss = net(title, abstract, t_mask, a_mask, indice)
    loss.backward()
    de_trainer.step(batch_size = batchsize)
    en_trainer.step(batch_size = batchsize)
    total_loss += loss.mean().asscalar()
    if (index+1)%100000 == 0:
        avg_loss = total_loss/(100000)
        valida_score = validation()
        print('epoch %d, avg loss %.4f, time %.2f' %(
        epoch, avg_loss, time()-tic))
        epoch = epoch+1
        total_loss = 0
        
        valida_loss = valida(validaset)
        print('epoch %d, validation loss: %.4f' %(
        epoch,  valida_loss))
    if epcoh >4:
        filenname = 'epoch'+'net.params'
        net.save_parameters(filename)
        
        
    
    
    
    
