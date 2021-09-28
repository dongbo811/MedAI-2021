import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
import cv2
from medpy import metric
if __name__ == '__main__':
    for id in range(1, 6):
        id = str(id)
        parser = argparse.ArgumentParser()
        parser.add_argument('--testsize', type=int, default=352, help='testing size')
        parser.add_argument('--pth_path', type=str, default='./model_pth/' + id + '_PolypPVT/PolypPVT.pth')
        opt = parser.parse_args()
        model = PolypPVT()
        model.load_state_dict(torch.load(opt.pth_path))
        model.cuda()
        model.eval()
        PA = 0.0
        dice = 0.0
        JS = 0.0
        for _data_name in ['KP']:

            ##### put data_path here #####
            data_path = '/home/hhedeeplearning/share/dongbo/CODE/DATASET/uuUNet_raw/KS/' + id +'_test/'
            ##### save_path #####
            # save_path = './result_map/PolypPVT/result1/'

            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            image_root = '{}/images/'.format(data_path)
            gt_root = '{}/masks/'.format(data_path)
            num1 = len(os.listdir(gt_root))
            test_loader = test_dataset(image_root, gt_root, 352)
            num1 = len(os.listdir(gt_root))
            DSC = 0.0
            for i in range(num1):
                image, gt, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                P1,P2 = model(image)
                res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                pred = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # mdepy
                pred = pred > 0.5
                h, w = pred.shape
                total = h * w
                pred_gt = (pred == gt)
                correct = (np.sum(pred_gt != 0))
                PA += correct / total
                # print(PA)
                dice += metric.dc(pred, gt)
                JS += metric.jc(pred, gt)

            ############output###############
            print('------------------' ,id, '------------------' )
            print(id, "Dice", dice / num1)
            print(id, "JS", JS / num1)
            print(id, "PA", PA / num1)
