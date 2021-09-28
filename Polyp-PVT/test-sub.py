# import torch
# import torch.nn.functional as F
# import numpy as np
# import os, argparse
# from scipy import misc
# from lib.pvt import PolypPVT
# from utils.dataloader import test_dataset
# import cv2

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--testsize', type=int, default=352, help='testing size')
#     parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT/13PolypPVT-best.pth')
#     opt = parser.parse_args()
#     model = PolypPVT()
#     model.load_state_dict(torch.load(opt.pth_path))
#     model.cuda()
#     model.eval()
#     for _data_name in ['KS']:

#         ##### put data_path here #####
#         data_path = '/home/hhedeeplearning/share/dongbo/CODE/DATASET/uuUNet_raw/{}/test/'.format(_data_name)
#         ##### save_path #####
#         save_path = './result_map/PolypPVT/{}/'.format(_data_name)

#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         image_root = '{}/images/'.format(data_path)
#         gt_root = '{}/masks/'.format(data_path)
#         num1 = len(os.listdir(gt_root))
#         test_loader = test_dataset(image_root, gt_root, 352)
#         num1 = len(os.listdir(gt_root))
#         DSC = 0.0
#         for i in range(num1):
#             image, gt, name = test_loader.load_data()
#             gt = np.asarray(gt, np.float32)
#             gt /= (gt.max() + 1e-8)
#             image = image.cuda()
#             P1,P2 = model(image)
#             res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)[0,0]
#             pred = res
#             pred[torch.where(pred>0)] /= (pred>0).float().mean()
#             pred[torch.where(pred<0)] /= (pred<0).float().mean()
#             pred = pred.sigmoid().data.cpu().numpy()
#             res = pred
#             # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#             cv2.imwrite(save_path+name, res*255)
#             input = res
#             target = np.array(gt)
#             N = gt.shape
#             smooth = 1
#             input_flat = np.reshape(input, (-1))
#             target_flat = np.reshape(target, (-1))
#             intersection = (input_flat * target_flat)
#             dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
#             dice = '{:.4f}'.format(dice)
#             dice = float(dice)
#             DSC = DSC + dice
#         print(DSC/num1)
#         print(_data_name, 'Finish!')

import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
# from utils.dataloader import test_dataset
import cv2
from medpy import metric
# from lib.Network_Res2Net_GRA_NCD import Network
import torch

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch


class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        #self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        #self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        im = image
        image = self.transform(image).unsqueeze(0)
        #gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.jpg'
        self.index += 1
        return image, im, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

if __name__ == '__main__':
    name = 'Polyp-PVT-main'
    for id in range(1, 6):
        id = str(id)
        parser = argparse.ArgumentParser()
        parser.add_argument('--testsize', type=int, default=352, help='testing size')
        parser.add_argument('--pth_path', type=str, default='/home/hhedeeplearning/share/dongbo/DATA/Kvasir-SEG/' + name + '/model_pth/' + id + '_PolypPVT/PolypPVT.pth')
        opt = parser.parse_args()
        model = PolypPVT()
        model.load_state_dict(torch.load(opt.pth_path))
        model.cuda()
        model.eval()
        for _data_name in ['KS-T1']:

            ##### put data_path here #####
            data_path = '/home/hhedeeplearning/share/dongbo/DATA/Kvasir-SEG/com/KS-T/'
            ##### save_path #####
            save_path = '/home/hhedeeplearning/share/dongbo/DATA/Kvasir-SEG/' + name + '/result_map/' + id + '_PolypPVT-T/masks/'

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image_root = '{}/'.format(data_path)
            gt_root = '{}/'.format(data_path)
            num1 = len(os.listdir(gt_root))
            test_loader = test_dataset(image_root, 352)
            num1 = len(os.listdir(gt_root))
            DSC = 0.0
            for i in range(num1):
                image, im, name1 = test_loader.load_data()
                
                image = image.cuda()
                res= model(image)
                h, w = im.size
                res = F.upsample(res[0] + res[1], size=[w,h], mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                cv2.imwrite(save_path+name1, res*255)
                
            #print(DSC/num1)
            print(_data_name, 'Finish!')

