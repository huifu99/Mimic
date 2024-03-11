import os
import sys
import copy
import random
import math
import numpy as np
import cv2
import torch
import torchvision
from tqdm import tqdm
from .config import cfg as spectre_cfg
from .src.spectre import SPECTRE


class VisualMesh():
    def __init__(self, cfg) -> None:
        # model
        self.cfg = cfg
        self.device = cfg['trainer']['device']
        spectre_cfg.pretrained_modelpath = "external/spectre/pretrained/spectre_model.tar"
        spectre_cfg.model.use_tex = False
        self.spectre = SPECTRE(spectre_cfg, device=self.device)
        self.spectre.eval()

    def forward(self, exp_out, exp, data_info):
        'input: expression coefficients'
        'output: mesh'
        n = self.cfg['trainer']['visual_images']
        if self.cfg['datasets']['dataset']=='mead':
            # template
            codedict = {}
            codedict['pose'] = torch.zeros((n*3, 6), dtype=torch.float).to(self.device)
            codedict['exp'] = torch.zeros((n*3, 50), dtype=torch.float).to(self.device)
            codedict['shape'] = torch.zeros((n*3, 100), dtype=torch.float).to(self.device)
            codedict['tex'] = torch.zeros((n*3, 50), dtype=torch.float).to(self.device)
            codedict['cam'] = torch.zeros((n*3, 3), dtype=torch.float).to(self.device)
            self.codedict = codedict
            # true coefficients
            coefficient_path = os.path.join(data_info, 'crop_head_info.npy')
            coefficient_info = np.load(coefficient_path, allow_pickle=True).item()['face3d_encode']
            coefficients = get_coefficients(coefficient_info)
            for key in coefficients:
                coefficients[key] = torch.FloatTensor(coefficients[key]).to(self.device)
            start_vis = random.randint(0,exp.shape[1]-1-n)  # 起始帧

            self.codedict['exp'][0:n] = exp_out[0, start_vis: start_vis+n,:-3]  # 生成的参数在平均脸上
            self.codedict['exp'][n:2*n] = exp[0, start_vis: start_vis+n,:-3]  # ground_truth参数在平均脸上
            self.codedict['exp'][2*n:3*n] = coefficients['exp'][start_vis: start_vis+n,:]  # ground_truth在原脸上

            self.codedict['pose'][0:n, 3:] = exp_out[0, start_vis: start_vis+n,-3:] # jaw pose
            self.codedict['pose'][n:2*n, 3:] = exp[0, start_vis: start_vis+n,-3:]

            self.codedict['cam'][0:n] = coefficients['cam'][start_vis: start_vis+n, :]  # 取n帧的cam
            self.codedict['cam'][n:2*n] = coefficients['cam'][start_vis: start_vis+n, :]
            self.codedict['cam'][2*n:3*n] = coefficients['cam'][start_vis: start_vis+n, :]

            self.codedict['pose'][2*n:3*n] = coefficients['pose'][start_vis: start_vis+n, :]  # 取n帧的pose
            self.codedict['shape'][2*n:3*n] = coefficients['shape'][start_vis: start_vis+n, :]  # # 取n帧的shape

        elif self.cfg['datasets']['dataset']=='mote':
            # template
            codedict = {}
            codedict['pose'] = torch.zeros((n*2, 6), dtype=torch.float).to(self.device)
            codedict['exp'] = torch.zeros((n*2, 50), dtype=torch.float).to(self.device)
            codedict['shape'] = torch.zeros((n*2, 100), dtype=torch.float).to(self.device)
            codedict['tex'] = torch.zeros((n*2, 50), dtype=torch.float).to(self.device)
            codedict['cam'] = torch.zeros((n*2, 3), dtype=torch.float).to(self.device)
            self.codedict = codedict
            # true coefficients
            # coefficient_path = os.path.join(self.cfg['datasets']['data_root'], data_info[0][0], data_info[1][0], 'train1_all.npz')
            # coefficient_info = np.load(coefficient_path, allow_pickle=True)['face'][-1*self.cfg['datasets']['eval_frames']:, :]
            # coefficients = get_coefficients(coefficient_info)
            # for key in coefficients:
            #     coefficients[key] = torch.FloatTensor(coefficients[key]).to(self.device)
            start_vis = random.randint(0,exp.shape[1]-1-n)  # 起始帧

            self.codedict['exp'][0:n] = exp_out[0, start_vis: start_vis+n,:-3]  # 生成的参数在平均脸上
            self.codedict['exp'][n:2*n] = exp[0, start_vis: start_vis+n,:-3]  # ground_truth参数在平均脸上

            self.codedict['pose'][0:n, 3:] = exp_out[0, start_vis: start_vis+n,-3:] # jaw pose
            self.codedict['pose'][n:2*n, 3:] = exp[0, start_vis: start_vis+n,-3:]
            
            cam = torch.tensor([8.8093824, 0.00314824, 0.043486204]).unsqueeze(0).repeat(n, 1)  # cam
            self.codedict['cam'][0:n] = cam
            self.codedict['cam'][n:2*n] = cam

        opdict = self.spectre.decode(self.codedict, rendering=True, vis_lmk=False, return_vis=False)
        # rendered_images = torchvision.utils.make_grid(opdict['rendered_images'].detach().cpu(), nrow=n)
        return opdict['rendered_images']

    def infer(self, exp, cfg, exp_gt=None, render_batch=100):
        'input: expression coefficients'
        'output: mesh'
        n = exp.shape[1]
        if self.cfg['datasets']['dataset']=='mead':
            coefficient_path = os.path.join(cfg['datasets']['data_root'], cfg['test']['audio_path']).replace('audio.wav', 'crop_head_info.npy')
            coefficient_info = np.load(coefficient_path, allow_pickle=True).item()['face3d_encode']
            coefficients = get_coefficients(coefficient_info)
            assert exp.shape[1]==coefficients['exp'].shape[0]
            coefficients_pred = copy.deepcopy(coefficients)
            for key in coefficients:
                coefficients[key] = torch.FloatTensor(torch.from_numpy(coefficients[key])).to(self.device)
                coefficients_pred[key] = torch.FloatTensor(torch.from_numpy(coefficients_pred[key])).to(self.device)
                if key == 'exp':
                    coefficients_pred[key] = exp[0][:, :-3]
                elif key == 'pose':
                    coefficients_pred[key][:, -3:] = exp[0][:, -3:]
        elif self.cfg['datasets']['dataset']=='mote':
            coefficients = {}
            coefficients['pose'] = torch.zeros((n, 6), dtype=torch.float).to(self.device)
            coefficients['exp'] = torch.zeros((n, 50), dtype=torch.float).to(self.device)
            coefficients['shape'] = torch.zeros((n, 100), dtype=torch.float).to(self.device)
            coefficients['tex'] = torch.zeros((n, 50), dtype=torch.float).to(self.device)
            coefficients['cam'] = torch.zeros((n, 3), dtype=torch.float).to(self.device)
            coefficients_pred = copy.deepcopy(coefficients)
            # cam = torch.tensor([8.8093824, 0.00314824, 0.043486204]).unsqueeze(0).repeat(n, 1).to(self.device)  # cam
            cam = torch.tensor([8.740263, -0.00034628902, 0.020510273]).unsqueeze(0).repeat(n, 1).to(self.device)  # cam
            for key in coefficients:
                # coefficients[key] = torch.FloatTensor(torch.from_numpy(coefficients[key])).to(self.device)
                # coefficients_pred[key] = torch.FloatTensor(torch.from_numpy(coefficients_pred[key])).to(self.device)
                if key == 'exp':
                    if exp_gt is not None:
                        coefficients[key] = exp_gt[0][:, :-3]
                    coefficients_pred[key] = exp[0][:, :-3]
                elif key == 'pose':
                    if exp_gt is not None:
                        coefficients[key][:, -3:] = exp_gt[0][:, -3:]
                    coefficients_pred[key][:, -3:] = exp[0][:, -3:]
                elif key == 'cam':
                    coefficients[key] = cam[:, :]
                    coefficients_pred[key] = cam[:, :] 
        n_batch = int(math.ceil(n/render_batch))
        rendered_images, rendered_images_pred = [], []
        for i in range(n_batch):
            coefficients_render, coefficients_pred_render = {}, {}
            for k in coefficients:
                start_f, end_f = i*render_batch, min((i+1)*render_batch, n)
                coefficients_render[k] = coefficients[k][start_f: end_f]
                coefficients_pred_render[k] = coefficients_pred[k][start_f: end_f]

            if exp_gt is not None:
                opdict = self.spectre.decode(coefficients_render, rendering=True, vis_lmk=False, return_vis=False)
                rendered_images.append(opdict['rendered_images'].detach().cpu())
            opdict_pred = self.spectre.decode(coefficients_pred_render, rendering=True, vis_lmk=False, return_vis=False)
            rendered_images_pred.append(opdict_pred['rendered_images'].detach().cpu())
        if exp_gt is not None:
            rendered_images_cat = torch.cat(rendered_images, dim=0)
        else:
            rendered_images_cat = None
        rendered_images_pred_cat = torch.cat(rendered_images_pred, dim=0)

        return rendered_images_cat, rendered_images_pred_cat
        # opdict = self.spectre.decode(coefficients, rendering=True, vis_lmk=False, return_vis=False)
        # opdict_pred = self.spectre.decode(coefficients_pred, rendering=True, vis_lmk=False, return_vis=False)
        # return opdict['rendered_images'], opdict_pred['rendered_images']

    def exp2mesh(self, coefficients_info, pose0=True, render_batch=100):
        n = coefficients_info.shape[0]
        if coefficients_info.shape[-1] == 53:
            coefficients = {}
            coefficients['pose'] = torch.zeros((n, 6), dtype=torch.float).to(self.device)
            coefficients['exp'] = torch.zeros((n, 50), dtype=torch.float).to(self.device)
            coefficients['shape'] = torch.zeros((n, 100), dtype=torch.float).to(self.device)
            coefficients['tex'] = torch.zeros((n, 50), dtype=torch.float).to(self.device)
            coefficients['cam'] = torch.zeros((n, 3), dtype=torch.float).to(self.device)
            cam = torch.tensor([8.740263, -0.00034628902, 0.020510273]).unsqueeze(0).repeat(n, 1).to(self.device)  # cam
            for key in coefficients:
                # coefficients[key] = torch.FloatTensor(torch.from_numpy(coefficients[key])).to(self.device)
                # coefficients_pred[key] = torch.FloatTensor(torch.from_numpy(coefficients_pred[key])).to(self.device)
                if key == 'exp':
                    coefficients[key] = coefficients_info[:, :-3]
                elif key == 'pose':
                    coefficients[key][:, -3:] = coefficients_info[:, -3:]
                elif key == 'cam':
                    coefficients[key] = cam[:, :]
        elif coefficients_info.shape[-1] == 209 or coefficients_info.shape[-1] == 213 or coefficients_info.shape[-1] == 236:
            coefficients = get_coefficients(coefficients_info)
            cam = torch.tensor([8.740263, -0.00034628902, 0.020510273]).unsqueeze(0).repeat(n, 1).to(self.device)  # cam
            for key in coefficients:
                coefficients[key] = torch.FloatTensor(coefficients[key]).to(self.device)
                if pose0:
                    if key == 'pose':
                        coefficients[key][:, :3] = torch.zeros_like(coefficients[key][:, :3])
                    elif key == 'shape' or key == 'tex':
                        coefficients[key] = torch.zeros_like(coefficients[key])
                    elif key == 'cam':
                        coefficients[key] = cam

        n_batch = int(math.ceil(n/render_batch))
        rendered_images = []
        vertices = []
        for i in tqdm(range(n_batch)):
            coefficients_batch = {}
            for k in coefficients:
                start_f, end_f = i*render_batch, min((i+1)*render_batch, n)
                coefficients_batch[k] = coefficients[k][start_f: end_f]
            opdict = self.spectre.decode(coefficients_batch, rendering=True, vis_lmk=False, return_vis=False)
            rendered_images.append(opdict['rendered_images'].detach().cpu())
            vertices.append(opdict['verts'].detach().cpu())
        rendered_images_cat = torch.cat(rendered_images, dim=0)
        vertices_cat = torch.cat(vertices, dim=0)
        return rendered_images_cat, vertices_cat


def get_coefficients(coefficient_info):
    coefficient_dict = {}
    coefficient_dict['pose'] = coefficient_info[:, :6]
    coefficient_dict['exp'] = coefficient_info[:, 6:56]
    coefficient_dict['shape'] = coefficient_info[:, 56:156]
    coefficient_dict['tex'] = coefficient_info[:, 156:206]
    coefficient_dict['cam'] = coefficient_info[:, 206:209]
    # coefficient_dict['light'] = coefficient_info[:, 209:236]
    return coefficient_dict
