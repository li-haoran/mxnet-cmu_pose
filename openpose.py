import mxnet as mx
from mod import poseMod
from config import config
from MPII import MPII
from dataIter import POSEIter
from mod import testMod

def train():
    batch_size=12
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
    pose_mod.init_optimizer(optim='sgd',lr=2e-5,lr_schedule=mx.lr_scheduler.FactorScheduler(step*10,0.333))
    pose_mod.fit(dataiter,100)

def test():
    import matplotlib.pyplot as plt
    import numpy as np
    batch_size=1
    ctx=mx.gpu(0)
    image_set='/home/lihaoran/MPII/images'
    annotation_file='/home/lihaoran/MPII/MPI_annotations.json'
    cache_path='/home/lihaoran/open_pose/data'
    openpose_params='/home/lihaoran/open_pose/model/OpenPose_vgg19-0001.params'

    mpii=MPII(image_set,annotation_file,cache_path)
    poses=mpii.gt_roidb()

    dataiter=POSEIter(poses,batch_size,ctx=ctx)

    pose_mod=testMod(dataiter.provide_data,dataiter.provide_label,ctx,openpose_params,batch_size)
    for i,batch in enumerate(dataiter):
        img=batch.data[0]
        heatmaps=batch.label[1]
        paf,heatmap=pose_mod.predict(img)
        fig,axes=plt.subplots(4,4)
        axra=axes.ravel()
        for i in range(heatmap[5].shape[1]):
            axra[i].imshow(heatmap[5][0,i,:,:])
            axra[i].axis('off')
        axra[14].imshow(np.max(heatmaps.asnumpy()[0],axis=0))
        axra[15].imshow(np.uint8(((img.asnumpy()[0]).transpose((1,2,0))+0.5)*256))
        plt.show()
        plt.close('all')


if __name__ =='__main__':
    #test()
    train()
