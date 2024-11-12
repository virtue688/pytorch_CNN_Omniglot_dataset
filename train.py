import os.path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.dataloader_cl import Dataset, dataset_collate
from utils.trainer import fit_one_epoch

from nets.model import Baseline


if __name__ == "__main__":
    Cuda = False  #使用CPU为False，GPU为True
    # ------------------------------------------------------#
    #   pretrained_model_path        网络预训练权重文件路径
    # ------------------------------------------------------#
    pretrained_model_path  = ''
    # ------------------------------------------------------#
    #   input_shape     输入的shape大小
    # ------------------------------------------------------#
    input_shape = [28, 28]
    batch_size = 32
    Init_Epoch = 0
    Epoch = 150

    # ------------------------------------------------------#
    #   Init_lr     初始学习率
    # ------------------------------------------------------#
    Init_lr = 0.001
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 5
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'logs/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_workers = 0

    # ------------------------------------------------------#
    #   train_val_dataset_path   训练和测试文件路径
    # ------------------------------------------------------#
    train_val_dataset_path = 'dataset/NewDataset.mat'

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------#
    #   创建模型
    # ------------------------------------------------------#
    model = Baseline()

    if pretrained_model_path != '':
        print('Load weights {}.'.format(pretrained_model_path))
        pretrained_dict = torch.load(pretrained_model_path, map_location= device)
        model.load_state_dict(pretrained_dict)

    model_train = model.train()

    if Cuda:
        Generator_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        Generator_train = Generator_train.cuda()

    opt_model = torch.optim.Adam(model.parameters(), lr=Init_lr)

    # ---------------------------------------#
    #   构建数据集加载器。
    # ---------------------------------------#
    train_dataset = Dataset(train_val_dataset_path, input_shape, epoch_length=Epoch, is_train=True)
    val_dataset = Dataset(train_val_dataset_path, input_shape, epoch_length=Epoch, is_train=False)

    shuffle = True

    train_gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=None)

    val_gen = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=None)

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#
    for epoch in range(Init_Epoch, Epoch):
        epoch_step = train_dataset.length // batch_size
        epoch_step_val = val_dataset.length // batch_size
        train_gen.dataset.epoch_now = epoch
        val_gen.dataset.epoch_now = epoch

        fit_one_epoch(model_train, model, opt_model, epoch, epoch_step, epoch_step_val, train_gen, val_gen, Epoch, Cuda, save_period, save_dir)