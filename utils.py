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
    max_length = get_length(batch).max().asnumpy()[0]
    return batch[:,:int(max_length)]

def grad_clipping(params, theta, ctx):
    norm = nd.array([0.0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
