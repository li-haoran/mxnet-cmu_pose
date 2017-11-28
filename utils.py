import numpy as np
import cv2
from config import config

def random_rotate_scale(img,angle=config.ANGLE_RANGE,scale_range=config.SCALE_RANGE,rgb_mean=config.RGB_MEAN):
    h,w=img.shape[:-1]
    ag=np.random.randint(angle[0],angle[1],1)[0]
    range=scale_range[1]-scale_range[0]
    sc=np.random.rand(1)[0]*range+scale_range[0]
    affine_matrix=cv2.getRotationMatrix2D((w/2,h/2),ag,sc)
    affine_img=cv2.warpAffine(img, affine_matrix, (w, h),cv2.BORDER_CONSTANT,
                                 borderValue=list(rgb_mean))
    return affine_img,affine_matrix

def random_crop(img,crop_size=config.CROP_SIZE,random_bound=config.RANDOM_BOUND,rgb_mean=config.RGB_MEAN):
    h,w=img.shape[:-1]   
    h_start=np.random.randint(-random_bound,random_bound,1)[0]
    w_start=np.random.randint(-random_bound,random_bound,1)[0]

    h_end=h_start+crop_size[0]
    w_end=w_start+crop_size[1]

    pad_h_up=max(0-h_start,0)
    pad_h_bottom=max(h_end-h,0)

    pad_w_left=max(0-w_start,0)
    pad_w_right=max(w_end-w,0)
    img=cv2.copyMakeBorder(img,pad_h_up,pad_h_bottom,pad_w_left,pad_w_right,cv2.BORDER_CONSTANT,
                                 value=list(rgb_mean))
    h0=h_start+pad_h_up
    h1=h0+crop_size[0]
    w0=w_start+pad_w_left
    w1=w0+crop_size[1]

    img=img[h0:h1,w0:w1,:]

    return img,(w_start,h_start)

    
def random_noise(img,noise=config.NOISE_VALUE):
    h,w=img.shape[:-1]
    noise=np.random.rand(h,w,1)*noise-noise/2
    img=img+noise
    return img

def random_brighten(img,brighten=config.BRIGHTEN):
    value=np.random.randint(0,brighten,1)[0]-brighten/2
    img+=value
    return img

def do_flip(img,points):
    '''
    here the img is [c,h,w]
    point is [n,14,2]
    '''
    h,w,c=img.shape
    num_pose,npart,xy=points.shape
    if num_pose>0:
        points[:,:,0]=w-1-points[:,:,0]
        if config.SWAP_LEFT_RIGHT:
            temp=points[:,config.LEFT,:].copy()
            points[:,config.LEFT,:]=points[:,config.RIGHT,:]
            points[:,config.RIGHT,:]=temp
    img=img[:,::-1,:]
    return img,points


def Image_augment(img_path,data_size=config.DATA_SIZE):
    img=cv2.imread(img_path)
    img=img[:,:,[2,1,0]]## 
    img=img.astype(np.float32)
    h,w=img.shape[:-1]
    img=cv2.resize(img,tuple(data_size),interpolation=cv2.INTER_LINEAR)
    scale_x=data_size[1]*1.0/w
    scale_y=data_size[0]*1.0/h
    img,mat=random_rotate_scale(img)
    img,place=random_crop(img)
    mat[0,2]-=place[0]
    mat[1,2]-=place[1]
    mat[:,0]*=scale_x
    mat[:,1]*=scale_y
    #img=random_noise(img)

    return img,mat

def visual(img,points):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.patches import Circle
    eps=1e-14

    link_num=len(config.PAIR_CONFIGS)/2
    colors=[]
    for link in range(link_num):
        colors.append(np.random.rand(3))

    num_pose,npart,xy=points.shape
    fig, ax= plt.subplots(1,1,subplot_kw={'aspect': 'equal'})
    ax.imshow(np.uint8(img/260*255))

    for ai in range(num_pose):    
        pose=points[ai,:,:]
        for i in range(link_num):
            link=(config.PAIR_CONFIGS[i*2],config.PAIR_CONFIGS[i*2+1])
            start = pose[link[0],:]
            end = pose[link[1],:]
            if 1-np.all(start>0) or 1-np.all(end>0) or 1-np.all(end<368) or 1-np.all(end<368):
                continue            
            color=(0,0,0)
            circle=Circle(start,2,fill=True,facecolor=color)
            ax.add_artist(circle)      
            circle=Circle(end,2,fill=True,facecolor=color)
            ax.add_artist(circle)

            center= (start+end)/2
            width = np.sqrt(np.sum((end-start)**2))
            height = 0.15 *width
            tan= (end[1]-start[1])/(end[0]-start[0]+eps)
            angle=np.arctan(tan)/np.pi *180
            stem=Ellipse(center,width=width,height=height,angle=angle,color=colors[i],alpha=0.5)
            ax.add_artist(stem)


        ax.axis('off')

    plt.show()

if __name__=='__main__':
    path=r'D:\dataset\evalMPII\mpii_human_pose_v1\images\082062225.jpg'
    #img,mat=Image_augment(path)
    point=np.array([960,540,1]).reshape((3,1))
    point2=np.array([540,540,1]).reshape((3,1))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fg,ax=plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            img,mat=Image_augment(path)
            new_point=mat.dot(point)
            new_point2=mat.dot(point2)
            ax[i,j].imshow(np.uint8(img))
            circle=Circle(new_point,10,fill=True,facecolor=(1,0,0))
            ax[i,j].add_artist(circle)
            circle=Circle(new_point2,10,fill=True,facecolor=(1,0,0))
            ax[i,j].add_artist(circle)
            ax[i,j].axis('off')
    plt.show()

