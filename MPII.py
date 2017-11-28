"""
Pascal VOC database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""

import cPickle
import cv2
import os
import numpy as np
import json
from config import config

class MPII(object):
    def __init__(self, image_set, annotation_file,cache_path):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """

        self.image_path = image_set
        self.cache_path= cache_path
        self.annotation = json.loads(open(annotation_file,'r').read())['root'] 
        #super(MPII, self).__init__('MPII_Train',image_set, annotation_file,cache_path)  # set self.name
        self.classes = ['__background__',  # always index 0
                        'person']
        self.num_classes = 2
        self.part_label=config.PART_LABELS
        self.index2index=[9,8,12,11,10,13,14,15,2,1,0,3,4,5] # for the dataset build from the mpii json to pkl
        self.num_parts= config.NUM_PARTS
        assert len(self.index2index)==self.num_parts,'support only this num_part'
        self.num_images = len(self.annotation)
        print 'num_images', self.num_images



    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['poses', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, 'MPII_OPEN_POSE_Train_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format('MPII', cache_file)
            return roidb

        gt_roidb = [self.load_MPII_annotation(index) for index in range(self.num_images)]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
   
    def load_MPII_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """

        
        roi_rec = dict()
        roi_rec['image'] = os.path.join(self.image_path,self.annotation[index]['img_paths'])
        #size = cv2.imread(roi_rec['image']).shape
        #roi_rec['height'] = size[0]
        #roi_rec['width'] = size[1]
        roi_rec['height'] = self.annotation[index]['img_height']
        roi_rec['width'] = self.annotation[index]['img_width']

        
        numOtherPerson=self.annotation[index]['numOtherPeople']
        otherPersonJoints=[]
        if numOtherPerson >0:
            if numOtherPerson>1:
                otherPersonJoints=otherPersonJoints+self.annotation[index]['joint_others']
            else:
                otherPersonJoints.append(self.annotation[index]['joint_others'])
        mainPersonJoints=self.annotation[index]['joint_self']
        allPerson=otherPersonJoints+[mainPersonJoints]
        num_objs = len(allPerson)

        poses = np.zeros((num_objs, self.num_parts*3), dtype=np.float32)
       
        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(allPerson):
            
            po=np.zeros((16,3),dtype=np.float32)
            po[0:len(obj),:]=np.array(obj,dtype=np.float32)
            assert po.shape[0] ==16,'the image is wrong'

            poses[ix, :] = po[self.index2index,:].ravel() ### obj must [14,3]

        roi_rec.update({'poses': poses,})
        return roi_rec

