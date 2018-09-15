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
    max_length = get_length(batch).max().asscalar()
    return batch[:,:int(max_length)]

def grad_clipping(params, theta, ctx):
    norm = nd.array([0.0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def generate_batch(t,a):
    t = (t!=0)
    max_t = int(t.sum(1).max().asscalar())
    a = (a!=0)
    max_a = int(a.sum(1).max().asscalar())
    return t[:,:max_t], a[:,:max_a] 
    
def unk(batch):
    mask = (batch<50000)
    indice = nd.where(mask, batch, 4*nd.ones_like(a))
    return indice

def dynamic_idx(instance):
    v = set(instance)
    oov = [idx for idx in v if idx>49999]
    v_to_oov={}
    for i in oov:
        v_to_oov[i] = len(v_to_oov)+49999
    for j in range(len(instance)):
        if instance[j]>49999:
            instance[j] = v_to_oov[instance[j]]
            
    return instance

def dynamic_idx_batch(batch):
    for index,_ in enumerate(batch):
        batch[index] = dynamic_idx(batch[index])
    return batch
