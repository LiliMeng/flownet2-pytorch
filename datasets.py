import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np

from glob import glob
import utils.frame_utils as frame_utils

from scipy.misc import imread, imresize
import scipy.misc
import cv2

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)/2:(self.h+self.th)/2, (self.w-self.tw)/2:(self.w+self.tw)/2,:]

class MpiSintel(data.Dataset):
    def __init__(self, args, img_list_file, is_cropped = True, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates
        self.flow_list = []
        self.image_list = []
        
        with open(img_list_file) as f:
            lines_img = f.readlines()

          # list for storing data and label for all the videos

        all_imgs_rgb_filenames = []
        labels = []
          
          
        for i in range(len(lines_img)):
            
            #print(len(all_imgs_rgb))
            label_img = int(lines_img[i].split('\t')[1])

           
            img_path_suffix = lines_img[i].split('\t')[0]
            img_names = os.listdir(img_path_suffix)


            start_idx = random.randint(1, (len(img_names)-3))
  
            img1 = os.path.join(img_path_suffix, 'image_' + str('%05d'%(start_idx)) + '.jpg')
            img2 = os.path.join(img_path_suffix, 'image_' + str('%05d'%(start_idx+2)) + '.jpg')
            
            # print("image names*******************")
            # print(img1)
            # print(img2)
            if not isfile(img1) or not isfile(img2):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [int(label_img)]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])/64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])/64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        # print("*********************************")
        # print(self.image_list[index][0])
        # print(self.image_list[index][1])


        img1 = cv2.resize(img1, dsize=(368, 276), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, dsize=(368, 276), interpolation=cv2.INTER_CUBIC)
        flow = self.flow_list[index]
        
        # scipy.misc.imsave('/home/lili/Video/flownet2-pytorch/tmp_data/img3.jpg', img1)
        # scipy.misc.imsave('/home/lili/Video/flownet2-pytorch/tmp_data/img4.jpg', img2)


        [h, w] = [276, 368]
        [th, tw] = [256, 256]  
        h1 = random.randint(0, h - th)
        w1 = random.randint(0, w - tw)
        
        # print("img1.shape before random crop")
        # print(img1.shape)

        # print("img2.shape before random crop")
        # print(img2.shape)
        img1 = img1[h1:(h1+th), w1:(w1+tw),:]
        img2 = img2[h1:(h1+th), w1:(w1+tw),:]

        # print("img1.shape")
        # print(img1.shape)

        # print("img2.shape")
        # print(img2.shape)
        #scipy.misc.imsave('/home/lili/Video/flownet2-pytorch/tmp_data/img5.jpg', img1)
        #scipy.misc.imsave('/home/lili/Video/flownet2-pytorch/tmp_data/img6.jpg', img2)
        images = [img1, img2]
       
        final_images = np.array(images).transpose(3,0,1,2)

        #print("images.shape before stack")
        #print(images.shape)
        #final_images= np.concatenate((images[:,0,:,:], images[:,1,:,:]), axis=0)
        
        #print("final_images.shape**************")
        #print(final_images.shape)
        final_images = torch.from_numpy(final_images.astype(np.float32))
      
        return [final_images], [flow]

    def __len__(self):
        return self.size * self.replicates

class MpiSintelClean(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'clean', replicates = replicates)

class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'final', replicates = replicates)

