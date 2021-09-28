import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
# from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
import cv2
from medpy import metric
from lib.Network_Res2Net_GRA_NCD import Network
import torch
if __name__ == '__main__':
    for id in range(1, 6):
        id = str(id)
        parser = argparse.ArgumentParser()
        parser.add_argument('--testsize', type=int, default=352, help='testing size')
        parser.add_argument('--pth_path', type=str, default='/home/zjudb/DATA2/KS-SIN-1/model_pth/' + id + '_PolypPVT/PolypPVT.pth')
        opt = parser.parse_args()
        model = Network()
        model.load_state_dict(torch.load(opt.pth_path))
        model.cuda()
        model.eval()
        dice = 0.0
        PA = 0.0
        JS = 0.0
        for _data_name in ['KS']:

            ##### put data_path here #####
            data_path = '/home/zjudb/DATA2/KS/' + id + '_test/'
            ##### save_path #####
            save_path = '/home/zjudb/DATA2/KS-SIN-1/result_map/' + id + '_PolypPVT/masks/'

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image_root = '{}/images/'.format(data_path)
            gt_root = '{}/masks/'.format(data_path)
            test_loader = test_dataset(image_root, gt_root, 352)
            num1 = len(os.listdir(gt_root))
            IOU = 0.0
            for i in range(num1):
                image, gt, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                res = model(image)
                res = F.upsample(res[0] + res[1] + res[2] + res[3], size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                pred = (res - res.min()) / (res.max() - res.min() + 1e-8)
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
