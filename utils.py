def get_length(x):
    x = (x!=0)
    length = x.sum(1)
    return length
