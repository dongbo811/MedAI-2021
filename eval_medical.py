import numpy as np
import os
import cv2 as cv
import SimpleITK as sitk
# from hausdorff import hausdorff_distance
from medpy import metric

img_path = r'E:\master\submit\Wangzheng\report1\result/try2a/'
gt_path =  r'E:\master\submit\Wangzheng\report1\result/masks2d2/'
imgs_path = os.listdir(img_path)
gts_path = os.listdir(gt_path)
num = len(gts_path)
###########pramenters#############
hd =0
AvgHD=0
HD=0
dice=0
JS = 0
VS = 0
hd95 = 0
asd = 0
assd = 0
precision = 0
recall = 0
sensitivity = 0
specificity = 0
true_negative_rate = 0
true_positive_rate = 0
ravd = 0
AvgHD = 0
PA = 0
##################################
for i in range(num):
    pred = cv.imread(os.path.join(img_path, gts_path[i]),0) / 255
    gt = cv.imread(os.path.join(gt_path, gts_path[i]),0) / 255

    # SITK
    mask = sitk.GetImageFromArray(gt, isVector=False)
    my_mask = sitk.GetImageFromArray(pred, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(mask > 0.5, my_mask > 0.5)
    AvgHD += hausdorffcomputer.GetAverageHausdorffDistance()

    # mdepy
    pred = pred >0.5
    h, w = pred.shape
    total = h*w
    pred_gt = (pred == gt)
    correct =(np.sum(pred_gt!=0))
    PA += correct/total
    # print(PA)
    dice+= metric.dc(pred,gt)
    JS += metric.jc(pred, gt)
    hd95 += metric.hd95(pred, gt, voxelspacing=1)
    hd += metric.hd(pred, gt, voxelspacing=0.15625)
    asd += metric.asd(pred, gt, voxelspacing=1)
    assd += metric.assd(pred, gt, voxelspacing=1)
    precision += metric.precision(pred, gt)
    recall += metric.recall(pred, gt)
    sensitivity += metric.sensitivity(pred, gt)
    specificity += metric.specificity(pred, gt)
    true_negative_rate += metric.true_negative_rate(pred, gt)
    true_positive_rate += metric.true_positive_rate(pred, gt)

############output###############
print("Dice", dice/num)
print("JS", JS/num)
print("PA", PA/num)
print("AvgHD", (AvgHD/num)*0.15625)
print("hd95", (hd95/num))
print("hd", (hd/num))
print("asd", (asd/num))
print("assd", (assd/num))
print("precision", (precision/num))
print("recall", (recall/num))
print("sensitivity", (sensitivity/num))
print("specificity", (specificity/num))
print("true_negative_rate", (true_negative_rate/num))
print("true_positive_rate", (true_positive_rate/num))
