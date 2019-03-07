import numpy as np
import scipy.special  
import seaborn as sns
from yapf.yapflib.yapf_api import FormatCode
from Jlearn import Jlearn
from functions import functions

prova=Jlearn(20)
prova2=functions(5,25)

J=np.loadtxt("J(srg,0.01,100).txt")
print(J)
