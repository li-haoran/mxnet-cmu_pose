import mxnet as mx
import numpy as np
from datetime import datetime
from sym import poseSymbol
from load_model import load_param
from metric import RMSEWithone
import logging

class poseMod(object):
    def __init__(self, data_shape,label_shape,batch_size,context):
        self.data_shape=data_shape
        self.label_shape=label_shape
        self.batch_size=batch_size
        self.context=context

        self.sym=poseSymbol(scale=1.0/self.batch_size)
        self.mod=mx.mod.Module(self.sym,data_names=['data','mask',],label_names=['paflabel','heatmaplabel',],context=self.context)
        self.mod.bind(data_shapes=self.data_shape,label_shapes=self.label_shape,for_training=True)


    def init_params(self,vgg19_prefix,epoch):
        vgg19_args,vgg19_auxs=load_param(vgg19_prefix,epoch,convert=True,ctx=self.context)
        arg_name=self.sym.list_arguments()
        new_args={}
        for k,v in vgg19_args.iteritems():
            if k in arg_name:
                new_args[k]=v

        print 'init the model from vgg 19',new_args.keys()

        #initializer=mx.init.Xavier(factor_type="in", magnitude=2.34)
        initializer=mx.init.Normal(sigma=0.01)
        self.mod.init_params(initializer=initializer,arg_params=new_args,allow_missing=True)
        print 'init finishing!'

    def init_optimizer(self,optim='sgd',lr=1e-6,lr_schedule=mx.lr_scheduler.FactorScheduler(10000,0.1)):
        if optim=='sgd':
            optimizer_params={ 'momentum':0.9,
                               'wd':  0.00005,
                               'learning_rate':lr,
                               'lr_scheduler':lr_schedule}
            print 'using sgd optimizer'
            print 'optimal prarams:',optimizer_params
        elif optim=='adam':
            optimizer_params={ 'beta1':0.9,
                               'wd':  0.00005,
                               'learning_rate':lr,
                               'lr_scheduler':lr_schedule}
            print 'using adam optimizer'
            print 'optimal prarams:',optimizer_params

        self.mod.init_optimizer(optimizer=optim,optimizer_params=optimizer_params)
        print 'init params successfully'

    def fit(self,dataIter,epoch):
        logging.basicConfig(level=logging.DEBUG)
        eval_metric=RMSEWithone()
        loss=[]
        def norm_stat(d):
            return mx.nd.norm(d)/np.sqrt(d.size)
        mon = mx.mon.Monitor(1, norm_stat, pattern=".*output|.*backward_.*", sort=True)
        #self.mod.install_monitor(mon)
        
        
        for ip in range(epoch):
            dataIter.reset()
            eval_metric.reset()
            for i,batch in enumerate(dataIter):
                #mon.tic()
                self.mod.forward(batch,is_train=True)
                output=self.mod.get_outputs()
                self.mod.backward()
                self.mod.update()
                eval_metric.update(output)
                #mon.toc_print()
                if (i+1)%5==0:
                    name,value=eval_metric.get()
                    eval_metric.reset()
                    loss.append(value.reshape(1,-1))
                    stamp =  datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
                    print '%s [%d] epoch [%d] batch training RMSE loss for all \nstage1_L1 =%f,\nstage2_L1 =%f,\nstage3_L1 =%f,\nstage4_L1 =%f,\nstage5_L1 =%f,\nstage6_L1 =%f'\
                        %(stamp,ip,i,value[0],value[2],value[4],value[6],value[8],value[10])
                    print 'stage1_L2 =%f,\nstage2_L2 =%f,\nstage3_L2 =%f,\nstage4_L2 =%f,\nstage5_L2 =%f,\nstage6_L2 =%f'\
                        %(value[1],value[3],value[5],value[7],value[9],value[11])

            if (ip+1)%1==0:
                self.mod.save_params('model/OpenPose_vgg19-%04d.params'%(ip))
                np.save('model/loss.npy',np.vstack(loss))



class testMod(object):
    def __init__(self, data_shape,label_shape,context,params_path,batch_size=1):
        self.data_shape=data_shape
        self.label_shape=label_shape
        self.batch_size=batch_size
        self.context=context

        self.sym=poseSymbol(train=False)
        self.mod=mx.mod.Module(self.sym,data_names=['data',],label_names=[],context=self.context)
        self.mod.bind(data_shapes=[self.data_shape[0],],for_training=False)
        self.mod.load_params(params_path)


    def predict(self,img):
        self.mod.forward(mx.io.DataBatch([img],[]),is_train=False)
        output=self.mod.get_outputs()
        paf=[output[i*2].asnumpy() for i in range(6)]
        heatmap=[output[i*2+1].asnumpy() for i in range(6)]
        return paf,heatmap
        





        
