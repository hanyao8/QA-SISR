import numpy as np 
import os

import pickle
import logging

from spams import trainDL

from config_train import config
from rnd_smp_patch import rnd_smp_patch
from patch_pruning import patch_pruning

dict_size   = config.dict_size         # dictionary size
lmbd        = config.lmbd          # sparsity regularization
patch_size  = config.patch_size            # image patch size
nSmp        = config.nSmp       # number of patches to sample
upscale     = config.upscale            # upscaling factor

train_img_path = 'data/train_hr/'   # Set your training images dir

################################################################################

log_format = "%(asctime)s | %(message)s"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(config.output_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("initialize logging")
logging.info("args = "+str(config))

################################################################################

# Randomly sample image patches
Xh, Xl = rnd_smp_patch(train_img_path, patch_size, nSmp, upscale)

# Prune patches with small variances
Xh, Xl = patch_pruning(Xh, Xl)
Xh = np.asfortranarray(Xh)
Xl = np.asfortranarray(Xl)

# Dictionary learning
logging.info("Learning Dh")
Dh = trainDL(Xh, K=dict_size, lambda1=lmbd, iter=100)

logging.info("Learning Dl")
Dl = trainDL(Xl, K=dict_size, lambda1=lmbd, iter=100)

# Saving dictionaries to files
with open('data/dicts/'+ 'Dh_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + config.exp_name + '.pkl', 'wb') as f:
    pickle.dump(Dh, f, pickle.HIGHEST_PROTOCOL)

with open('data/dicts/'+ 'Dl_' + str(dict_size) + '_US' + str(upscale) + '_L' + str(lmbd) + '_PS' + str(patch_size) + config.exp_name + '.pkl', 'wb') as f:
    pickle.dump(Dl, f, pickle.HIGHEST_PROTOCOL)
