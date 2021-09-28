from medpy import metric
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging




def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):
    # ema.apply_shadow()
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    IOU = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1, res2 = model(image)
        # eval iou
        res = F.upsample(res + res1 + res2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        gt = np.array(gt)
        res = res > 0.5
        IOU += metric.jc(res, gt)

    # ema.restore()
    return IOU / num1


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2, P3 = model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss = loss_P1 + loss_P2 + loss_P3
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ema.update()
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
    # save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # choose the best model

    global dict_plot

    test1path = '/home/zjudb/DATA2/KS/'
    if (epoch + 1) % 1 == 0:
        for dataset in  [id + '_test']:
            meandice = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, meandice))
            print(dataset, ': ', meandice)
            # dict_plot[dataset].append(meandice)

        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
            torch.save(model.state_dict(), save_path + str(epoch) + 'PolypPVT-best.pth')
            print('##############################################################################best', best)
            logging.info(
                '##############################################################################best:{}  || {}'.format(best, id ))


def plot_train(dict_plot=None, name=None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,
                     'ETIS-LaribPolypDB': 0.733, 'test': 0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()


if __name__ == '__main__':
    dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'PolypPVT'
    ###############################################
    for id in range(1, 6):
        id = str(id)
        parser = argparse.ArgumentParser()

        parser.add_argument('--epoch', type=int,
                            default=100, help='epoch number')

        parser.add_argument('--lr', type=float,
                            default=1e-4, help='learning rate')

        parser.add_argument('--optimizer', type=str,
                            default='AdamW', help='choosing optimizer AdamW or SGD')

        parser.add_argument('--augmentation',
                            default=False, help='choose to do random flip rotation')

        parser.add_argument('--batchsize', type=int,
                            default=10, help='training batch size')

        parser.add_argument('--trainsize', type=int,
                            default=352, help='training dataset size')

        parser.add_argument('--clip', type=float,
                            default=0.5, help='gradient clipping margin')

        parser.add_argument('--decay_rate', type=float,
                            default=0.1, help='decay rate of learning rate')

        parser.add_argument('--decay_epoch', type=int,
                            default=50, help='every n epochs decay learning rate')

        parser.add_argument('--train_path', type=str,
                            default='/home/zjudb/DATA2/KS/' + id + '_train/',
                            help='path to train dataset')

        parser.add_argument('--test_path', type=str,
                            default='/home/zjudb/DATA2/KS/' + id + '_test/',
                            help='path to testing Kvasir dataset')

        parser.add_argument('--train_save', type=str,
                            default='/home/zjudb/DATA2/PVT-KS1/model_pth/' + id + '_' + model_name + '/')

        opt = parser.parse_args()
        logging.basicConfig(filename='/home/zjudb/DATA2/PVT-KS1/' + id + '_train_log.log',
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

        # ---- build models ----
        # torch.cuda.set_device(0)  # set your gpu device
        model = PolypPVT().cuda()
        # ema = EMA(model, 0.999)
        # ema.register()

        best = 0

        params = model.parameters()

        if opt.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

        print(optimizer)
        image_root = '{}/images/'.format(opt.train_path)
        gt_root = '{}/masks/'.format(opt.train_path)

        train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                                  augmentation=opt.augmentation)
        total_step = len(train_loader)

        print("#" * 20, "Start Training", "#" * 20)

        for epoch in range(1, opt.epoch):
            adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
            train(train_loader, model, optimizer, epoch, opt.test_path)

        # plot the eval.png in the training stage
        # plot_train(dict_plot, name)
