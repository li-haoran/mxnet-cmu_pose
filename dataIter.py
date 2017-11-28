import numpy as np
import mxnet as mx
from utils import Image_augment
from utils import do_flip
from utils import visual
from config import config

class POSEIter(mx.io.DataIter):
    def __init__(self,annotations,batch_size,shuffle=True,ctx=mx.gpu(0)):
        self.annotations=annotations
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.length=len(annotations)
        self.index=range(self.length)
        self.data_names=['data',]
        self.label_names=['paflabel','heatmaplabel',]
        self.cur=0
        self.ctx=ctx
        self.reshape_mean=np.array(config.RGB_MEAN,dtype=np.float32).reshape((1,1,3))
        if self.shuffle:
            np.random.shuffle(self.index)
        self.next()
        self.reset()


    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_names, self.label)]

    def reset(self):
        self.cur=0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur+self.batch_size<=self.length

    def next(self):
        if self.iter_next():
            batch_ind=self.index[self.cur:self.cur+self.batch_size]
            self.cur+=self.batch_size
            self.get_batch(batch_ind)
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration


    def get_batch(self,batch_ind):

        img_list=[]
        pose_list=[]
        for id_in_this_batch,id in enumerate(batch_ind):
            img_path=self.annotations[id]['image']
            img,mat=Image_augment(img_path)
            
            #----------------------pose processing ------------------------
            poses_info=self.annotations[id]['poses']
            num_pose=poses_info.shape[0]
            poses_info=poses_info.reshape((num_pose,config.NUM_PARTS,3))

            pose=poses_info[:,:,:-1]
            pose=pose.reshape((num_pose*config.NUM_PARTS,2)).transpose((1,0))
            r_pose=np.vstack([pose,np.ones((1,num_pose*config.NUM_PARTS))])
            new_pose=mat.dot(r_pose)
            new_pose=new_pose.transpose((1,0))
            new_pose=new_pose.reshape((num_pose,config.NUM_PARTS,2))

            if config.DO_FLIP and np.random.randint(0,2,1)[0]:
                img,new_pose=do_flip(img,new_pose)

            #if config.VISUAL:
            #    visual(img,new_pose)

            #------------------------- pose/8.0 ---------------------------
            new_pose=new_pose/config.DS_SCALE
            #--------------------------------------------------------------
            visible=poses_info[:,:,-1].reshape((num_pose,config.NUM_PARTS,1))
            final_pose=np.concatenate([new_pose,visible],axis=2)
            assert final_pose.shape[2]==3 and final_pose.shape[1]==config.NUM_PARTS,'shape error'
            final_pose=final_pose.reshape((num_pose,config.NUM_PARTS*3))
            
            batch_pose=np.zeros((num_pose,config.NUM_PARTS*3+1),dtype = np.float32)
            batch_pose[:,0]=id_in_this_batch
            batch_pose[:,1:]=final_pose


            #-------------------img processing------------------------------
            img=img-self.reshape_mean#/256.0-0.5#
            img=img.transpose((2,0,1))
            img=np.expand_dims(img,axis=0)

            img_list.append(img)
            pose_list.append(batch_pose)
        self.data=[mx.nd.array(np.vstack(img_list),self.ctx),]
        
        pose=mx.nd.array(np.vstack(pose_list),self.ctx)
        paflabel=mx.nd.PARTAffineField(pose,output_shape=config.OUTPUT_SHAPE,pair_config=config.PAIR_CONFIGS,
                                       beam_width=config.BEAM_WIDTH,num_parts=config.NUM_PARTS,num_pairs=config.NUM_PAIRS,batch_size=self.batch_size)
        paflabel.wait_to_read()
        pose=mx.nd.array(np.vstack(pose_list),self.ctx)
        heatmaplabel=mx.nd.HEATMap(pose,output_shape=config.OUTPUT_SHAPE,sigma=config.SIGMA,num_parts=config.NUM_PARTS,batch_size=self.batch_size)
        heatmaplabel.wait_to_read()
        self.label=[paflabel,heatmaplabel]
        #if config.VISUAL:
        #    import matplotlib.pyplot as plt

        #    ht=heatmaplabel.asnumpy()
        #    for i in range(ht.shape[0]):
        #        hti=np.max(ht[i],axis=0)
        #        plt.imshow(hti)
        #        plt.show()





        
        
