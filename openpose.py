import mxnet as mx
from mod import poseMod
from config import config
from MPII import MPII
from dataIter import POSEIter

def train():
    batch_size=10
    ctx=mx.gpu(0)
    image_set='/home/lihaoran/MPII/images'
    annotation_file='/home/lihaoran/MPII/MPI_annotations.json'
    cache_path='/home/lihaoran/open_pose/data'
    vgg19_params='/home/lihaoran/open_pose/model/vgg19'

    mpii=MPII(image_set,annotation_file,cache_path)
    poses=mpii.gt_roidb()

    dataiter=POSEIter(poses,batch_size,ctx=ctx)

    pose_mod=poseMod(dataiter.provide_data,dataiter.provide_label,batch_size,ctx)
    pose_mod.init_params(vgg19_params,0)
    step=dataiter.length/batch_size
    pose_mod.init_optimizer(optim='adam',lr=2e-5,lr_schedule=mx.lr_scheduler.FactorScheduler(step*5,0.333))
    pose_mod.fit(dataiter,20)


if __name__ =='__main__':
    train()
