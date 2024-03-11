import os
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('.')
from external.spectre.src.utils import util
from external.spectre.src.utils.renderer import SRenderY, set_rasterizer


class Render(nn.Module):
    def __init__(self, image_size=224, background=True, device='cuda:0') -> None:
        super().__init__()
        self.image_size = image_size
        self.background = background
        self.device = device
        set_rasterizer('pytorch3d')
        obj_filename='external/spectre/data/head_template.obj'
        self.render = SRenderY(self.image_size, obj_filename=obj_filename, uv_size=256, rasterizer_type='pytorch3d').to(self.device)
    
    def forward(self, verts):
        if type(verts) is np.ndarray:
            verts = torch.FloatTensor(verts).to(self.device)
        n = verts.shape[0]
        cam = torch.tensor([8.740263, -0.00034628902, 0.020510273]).unsqueeze(0).repeat(n, 1).to(self.device)  # need try others
        trans_verts = util.batch_orth_proj(verts, cam);
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        h, w = self.image_size, self.image_size
        if self.background:
            background = None
        else:  # white
            background = torch.ones((n, 3, h, w)).to(self.device)
        shape_images, _, grid, alpha_images, pos_mask = self.render.render_shape(verts, trans_verts, h=h, w=w,
                                                                            images=background,
                                                                            return_grid=True,
                                                                            return_pos=True)
        rendered_images = self.postprocess(shape_images)
        return rendered_images
    
    def postprocess(self, rendered_images):
        # rendered_images (b, c, h, w)
        rendered_images = rendered_images.cpu().numpy()
        rendered_images = rendered_images*255.
        rendered_images = np.maximum(np.minimum(rendered_images, 255), 0)
        rendered_images = rendered_images.transpose(0,2,3,1)[:,:,:,[2,1,0]]
        rendered_images = rendered_images.astype(np.uint8).copy()
        return rendered_images


if __name__ == '__main__':
    import cv2
    render = Render()
    verts_path = '/root/autodl-tmp/data/fh/VOCASET/FaceFormer_processed/vertices_npy/FaceTalk_170725_00137_TA_sentence01.npy'
    save_root = os.path.join('tmp', 'FaceTalk_170725_00137_TA_sentence01')
    os.makedirs(save_root, exist_ok=True)
    verts = np.load(verts_path,allow_pickle=True)[::2,:]
    
    # verts_path = '/root/autodl-tmp/data/fh/HDTF/spectre_processed_25fps_16kHz/RD_Radio1_000/verts_new_shape1.npy'
    # save_root = os.path.join('tmp', 'RD_Radio1_000')
    # os.makedirs(save_root, exist_ok=True)
    # verts = np.load(verts_path, allow_pickle=True).item()['verts']
    verts = torch.FloatTensor(verts).to('cuda:0')
    for i in range(verts.shape[0]):
        vert = verts[i:i+1].view(-1, 5023, 3)
        render_img = render.forward(vert)
        img_save = os.path.join(save_root, str(i+1).zfill(6)+'.png')
        cv2.imwrite(img_save, render_img[0])