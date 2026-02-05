#%%
import time
import numpy as np
import torch
from torch import nn

# %%
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

#%%
class A:
    def __init__(self):
        self.b = 1

#%%
a = A()


# %%
@add_to_class(A)
def do(self):
    print(f'Class attribute "b" is {self.b}')

# %%
#a.do()

#%%
class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
    
#%%
class B(HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)