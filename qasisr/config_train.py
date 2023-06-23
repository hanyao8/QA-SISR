from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C
C.seed = 12345

############################################

C.dict_size   = 128         # dictionary size
C.lmbd        = 0.1          # sparsity regularization
C.patch_size  = 5            # image patch size
C.nSmp        = 100000       # number of patches to sample
C.upscale     = 3            # upscaling factor

############################################

C.root_dir = os.path.realpath(".")
print("root_dir:%s"%C.root_dir)

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.exp_name = 'exp_train_'+exp_time
C.output_dir = os.path.join(*[C.root_dir,'output',C.exp_name])
os.makedirs(C.output_dir, exist_ok=True)

print("exp_name: %s"%C.exp_name)
