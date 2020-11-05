from __future__ import absolute_import

import numpy as np
import time
from PIL import Image
from p_config import p_config
from region import Region
import cv2 as cv

from utils.viz import show_frame


class Tracker(object):

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic
    
    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    def track(self, img_files, box, seq_name, visualize=False):

        #sequence_dir = '/content/LT_DSE/progetto/' + seq_name + '/img/'
        gt_dir = '/content/LT_DSE/progetto/' + seq_name + '/groundtruth.txt'
        
        try:
            groundtruth = np.loadtxt(gt_dir, delimiter=',')
        except:
            groundtruth = np.loadtxt(gt_dir)

        region = Region(groundtruth[0, 0], groundtruth[0,1],groundtruth[0,2],groundtruth[0,3])
        frame_num = len(img_files)
        
        bBoxes = np.zeros((frame_num, 4))
        bBoxes[0, :] = box       

        times = np.zeros(frame_num)
        p=p_config()
        for f, img_file in enumerate(img_files):             
            
            image = cv.cvtColor(cv.imread(img_file), cv.COLOR_BGR2RGB)

            start_time = time.time()
            if f == 0:
                self.first_frame(image, region, video=seq_name, p=p, groundtruth=groundtruth)
                self.init(image, box)
            else:               
                region, confidence = self.update(image)
                bBoxes[f, :] = region
            times[f] = time.time() - start_time

            if visualize:
                show_frame(image, bBoxes[f, :])

        return bBoxes, times