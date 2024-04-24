import os
# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # single GPU for training
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
import datetime
import numpy as np
import copy
import cv2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from base.utilities import get_logger, get_parser, fixed_seed
"model"
from models.network import DisNetAutoregCycle as Model
"data"
from data.dataloader_HDTF import get_dataloaders
"loss"
from losses.loss_collections import ComposeCycleLoss as ComposeLoss
"val"
# from val import Val, ValTrainset


def main():
    fixed_seed(seed=42)
    cfg, params_dict = get_parser()
    device = 'cuda:0'
    output_dir = os.path.join(cfg.output, cfg.dataset+'_train{}_val-shape1'.format(str(cfg.train_ids)), cfg.exp_name)
    
    "log"
    logger = get_logger()
    log_dir = os.path.join(output_dir, 'log')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if cfg.continue_ckpt is None:
        writer = SummaryWriter(f'{log_dir}/runs/{timestamp}')

    "output"
    save_dir = os.path.join(output_dir, 'ckpt')
    os.makedirs(save_dir, exist_ok=True)

    "model"
    model = Model(cfg)
    if cfg.continue_ckpt is None:
        model = model.to(device)

    "data"
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']
    # val_loader = dataset['valid']

    "val"
    # val = Val(cfg, device)
    # val_trainset = ValTrainset(cfg, device)

    "optimizer"
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(cfg.base_lr), betas=(0.9, 0.999))

    "lr_scheduler"
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_sch_gamma)

    "loss"
    criterion = ComposeLoss(cfg).to(device)

    iteration = 0
    "continue train"
    if cfg.continue_ckpt is not None:
        ckpt_dict = torch.load(cfg.continue_ckpt, map_location='cpu')
        cfg.start_epoch = ckpt_dict['start_epoch']
        weights = ckpt_dict['model']
        model.load_state_dict(weights)
        model.to(device)
        optim = ckpt_dict['optimizer']
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=ckpt_dict['optimizer']['param_groups'][0]['lr'], betas=(0.9, 0.999))
        optimizer.load_state_dict(optim)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_sch_gamma, last_epoch=cfg.start_epoch-1)
        timestamp = '2023-07-10-23-27-58'
        writer = SummaryWriter(f'{log_dir}/runs/{timestamp}')
        iteration = cfg.start_epoch*len(train_loader)

    "train"
    for epoch in range(cfg.start_epoch, cfg.epochs):

        model.train()
        optimizer.zero_grad()
        if cfg.lr_sch_epoch is not None and (epoch+1) % cfg.lr_sch_epoch == 0:
             scheduler.step()
        loss_train_list = []
        for b, data in enumerate(train_loader):
            audio, vertices, template, one_hot, subject_id, init_state = data['audio'], data['vertices'], data['template'], data['one_hot'], data['subject_id'], data['init_state']
            vertices = vertices.to(device)
            label = vertices
            audio = audio.to(device)
            # text_label = text_label.to(device)
            template = template.to(device)
            one_hot = one_hot.to(device)
            id_label = torch.argmax(one_hot, dim=1)
            init_state = init_state.to(device)
            output = model(vertices, audio, template, init_state, id_label=id_label)
            # output = model(vertices, audio, template, one_hot)
            loss_dict = criterion(output, label, text_label=None, id_label=id_label)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train_list.append(loss.item())
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            log = "(Epoch {}/{}, Batch {}/{}), lr {}, TRAIN LOSS:{:.9f}".format((epoch+1), cfg.epochs, b+1, len(train_loader), lr, loss)
            logger.info(log)
            iteration += 1
            if iteration%cfg.print_freq==0:
                for loss_item in loss_dict:
                    if 'acc' not in loss_item:
                        writer.add_scalar('train/{}'.format(loss_item), loss_dict[loss_item].item(), iteration)
                # debug
                # if loss_dict['content_contrastive_loss'].item() > 4.5e-6:
                #     print(subject_id)
            # cal loss for epoch
            if b==0:
                loss_dict_epoch = {}
                for key in loss_dict:
                    loss_dict_epoch[key] = loss_dict[key]
            else:
                for key in loss_dict_epoch:
                    loss_dict_epoch[key] += loss_dict[key]

        for loss_item in loss_dict_epoch:
            writer.add_scalar('train/{}_epoch'.format(loss_item), loss_dict_epoch[loss_item].item()/(b+1), epoch+1)
        # writer.add_scalar('train/loss_epoch', np.mean(loss_train_list), epoch+1)
        
        if (epoch+1)%cfg.save_freq==0:
            state = {'params': params_dict['params'],
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'start_epoch': epoch+1,
                    }
            ckpt_dir = os.path.join(save_dir, 'Epoch_{}.pth'.format(epoch+1))
            torch.save(state, ckpt_dir)

        # if cfg.content_grl_loss.w_decay is not None and (epoch+1)%cfg.content_grl_loss.w_decay == 0:
        #     cfg.content_grl_loss.w = cfg.content_grl_loss.w*0.1
        # debug
        if cfg.content_grl_loss.w_decay is not None:
            if epoch+1 == 20:
                cfg.content_grl_loss.w = cfg.content_grl_loss.w*0.1
            elif epoch+1 == 40:
                cfg.content_grl_loss.w = cfg.content_grl_loss.w*0.5
                


if __name__ == '__main__':
    main()