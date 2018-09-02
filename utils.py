from mxnet import nd
def get_length(x):
    x = (x!=0)
    length = x.sum(1)
    return length

def return_mask(key,query):
    K = (key!=0).expand_dims(1)
    Q =(query!=0).expand_dims(-1)
    mask = nd.batch_dot(Q, K)
    return mask

def bucket(batch):
    max_length =int(get_length(batch).max().asnumpy[0])
    return batch[:,:max_length]
