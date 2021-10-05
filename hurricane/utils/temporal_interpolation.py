import numpy as np

def linear_interpolation(t0,d0,t1,d1,t):
    d = d0+(t-t0)*((d1-d0)/(t1-t0))
    return d


if __name__ == "__main__":
    d0 = [1,2,3,4,5]
    d1 = [100,200,300,400,500]
    d0 = np.array(d0).reshape(-1,1)
    d1 = np.array(d1).reshape(-1,1)
    t0 = 10
    t1 = 100
    t = 36
    d = linear_interpolation(t0,d0,t1,d1,t)
    print(d)