
# mxnet-cmu_pose  from version[cmu pose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
# Realtime Multi-Person Pose Estimation
[Zhe Cao](http://www.andrew.cmu.edu/user/zhecao), [Tomas Simon](http://www.cs.cmu.edu/~tsimon/), [Shih-En Wei](https://scholar.google.com/citations?user=sFQD3k4AAAAJ&hl=en), [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/).


### Network Architecture
![Teaser?](https://github.com/ZheC/Multi-Person-Pose-Estimation/blob/master/readme/arch.png)


### training processing

```python
batch_size=12
lr_schedule=mx.lr_scheduler.FactorScheduler(step*10,0.333)
if optim=='sgd':
    optimizer_params={ 'momentum':0.9,
                        'wd':  0.00005,
                        'learning_rate':lr,
                        'lr_scheduler':lr_schedule}
    print 'using sgd optimizer'
    print 'optimal prarams:',optimizer_params
```
### trainging speed
i didn't caculate, the general is 5~6 second 5 batches with batch size 12. i think it is similar to the original code.

### result showing
[x] loss during training

![plot?](https://github.com/li-haoran/mxnet-cmu_pose/blob/master/loss/open_pose.png)

[x] some samples from the dataset

![sample1?](https://github.com/li-haoran/mxnet-cmu_pose/blob/master/loss/sample1.png)

![sample1?](https://github.com/li-haoran/mxnet-cmu_pose/blob/master/loss/sample2.png)
## Thanks
[mxnet_Realtime_Multi-Person_Pose_Estimation](https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation) model_file ,and some others.

[mx-openpose](https://github.com/kohillyang/mx-openpose) mask generation and some others. 

## Citation
Please cite the paper in your publications if it helps your research:

    
    
    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
      }
	  
    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }
