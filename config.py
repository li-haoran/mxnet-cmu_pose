import numpy as np
from easydict import EasyDict as edict

config = edict()

## data augementations
config.CROP_SIZE = (368,368)
config.RANDOM_BOUND = 20
config.DATA_SIZE = (368,368)
config.ANGLE_RANGE = (-20,20)
config.SCALE_RANGE = (0.75,1.25)
config.NOISE_VALUE = 10
config.BRIGHTEN = 10
config.RGB_MEAN = (128,128,128)#(123.68, 116.779, 103.939)#
config.DO_FLIP=True
config.SWAP_LEFT_RIGHT=True
config.LEFT=(5,6,7,11,12,13)
config.RIGHT=(2,3,4,8,9,10)

config.MASK_TYPE='rect'#'polygon'#
config.VISUAL=True
## for paf and heatmap
config.OUTPUT_SHAPE=(46,46)
config.DS_SCALE=8.0
config.BEAM_WIDTH=0.2
config.SIGMA=2
config.NUM_PARTS=14
config.NUM_PAIRS=16
config.PART_LABELS=['head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri','Rhip', 'Rkne', 'Rank','Lhip', 'Lkne', 'Lank']
config.PAIR_CONFIGS=(0,1,1,2,2,3,3,4,1,5,5,6,6,7,2,8,5,11,8,11,8,9,9,10,11,12,12,13,1,8,1,11)

if __name__=='__main__':
    print config