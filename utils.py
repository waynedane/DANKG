def gen_mask(x):
    x = (x!=0)
    length = x.sum(1)
    return length
