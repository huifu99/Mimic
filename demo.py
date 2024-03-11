import os
import numpy as np
import cv2
import pickle
import shutil
import random
import librosa
import torch
from base.config import CfgNode
from transformers import Wav2Vec2Processor
from tools.render_spectre import Render

from models.network import DisNetAutoregCycle as Model

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import tempfile
from subprocess import call
from psbody.mesh import Mesh
from utils.render_pyrender import render_mesh_helper


class Infer():
    def __init__(self, ckpt, device='cuda:0') -> None:
        super().__init__()
        self.device = device
        ckpt_info = torch.load(ckpt, map_location='cpu')
        param_dict = ckpt_info['params']
        cfg = {}
        for key in param_dict:
            for k, v in param_dict[key].items():
                cfg[k] = v
        cfg = CfgNode(cfg)
        self.cfg = cfg

        self.fps = cfg.video_fps
        self.window = cfg.window
        self.motion_dim = cfg.motion_dim
        self.overlap = 10
        self.image_size = 224
        self.demo_path = 'demos'

        # load model
        weights = ckpt_info['model']
        self.model = Model(cfg)
        self.model.load_state_dict(weights)
        self.model.eval().to(device)

        # audio processor
        self.processor = Wav2Vec2Processor.from_pretrained(cfg.wav2vec2model)

        # render from spectre
        self.render = Render(image_size=self.image_size, background=True, device=device)
        
    
    def infer_from_wav(self, wav_file, style_ref, save_path, use_pytorch3d=True):
        os.makedirs(save_path, exist_ok=True)

        # load audio-huggingface
        speech_array, sampling_rate = librosa.load(wav_file, sr=16000)
        stride = int(sampling_rate//self.fps)
        audio_frame_num = speech_array.shape[0]//stride

        # load style reference
        id = style_ref.split('-')[0]
        video_name = style_ref.split('-')[1]
        style_npy = os.path.join(self.demo_path, 'style_ref', id, video_name, 'verts_new_shape1.npy')
        codedict = np.load(style_npy, allow_pickle=True).item()
        vertices = codedict['verts'].reshape(-1, self.motion_dim)
        vertices = torch.FloatTensor(vertices)  # (L, 15069)
        style_input_vertices = vertices[-self.window:].unsqueeze(0).to(self.device)

        # template
        template_pkl = os.path.join(self.demo_path, 'style_ref', id, style_ref+'.pkl')
        template = pickle.load(open(template_pkl, 'rb'))
        template = template.reshape(-1, self.motion_dim)
        template = torch.FloatTensor(template).to(self.device)

        # infer
        f = 0
        while f < audio_frame_num:
            if f == 0:
                init_state = torch.zeros((1, self.motion_dim)).to(self.device)
                temp_end_f = min(f+self.window, audio_frame_num)
            else:
                temp_end_f = min(f+self.window-self.overlap, audio_frame_num)
            temp_start_f = max(0, f-self.overlap)
            audio_batch = speech_array[temp_start_f*stride: temp_end_f*stride]
            audio_batch = np.squeeze(self.processor(audio_batch,sampling_rate=16000).input_values)
            audio_batch = torch.FloatTensor(audio_batch).to(self.device).unsqueeze(0)
            with torch.no_grad():
                out_batch, _ = self.model.predict(audio_batch, style_input_vertices, init_state, template)

            init_state = out_batch[:, -self.overlap, :]-template
            if f == 0:
                pred = out_batch.cpu()
            else:
                pred = torch.cat([pred, out_batch[:, self.overlap:, :].cpu()], dim=1)
            f = temp_end_f
        pred = pred.squeeze(0).numpy()
        pred = np.reshape(pred,(-1,self.motion_dim//3,3))

        # save verts
        wav_name = os.path.basename(wav_file).replace('.wav', '')
        file_name = "audio_{}-style_{}".format(wav_name, style_ref)
        save_npy = os.path.join(save_path, file_name+'.npy')
        os.makedirs(os.path.join(save_path), exist_ok=True)
        save_dict = {
            'verts_pred': pred,
            }
        np.save(save_npy, save_dict)
        print('Saved {}'.format(save_npy))

        # render
        num_frames = pred.shape[0]
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=save_path)
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.image_size, self.image_size), True)
        center = np.mean(pred[0], axis=0)
        for i_frame in range(num_frames):
            if use_pytorch3d:
                pred_img = self.render(pred[i_frame:i_frame+1])
                pred_img = pred_img[0]
            else:
                render_mesh = Mesh(pred[i_frame], self.template_mesh.f)
                pred_img = render_mesh_helper(self.cfg, render_mesh, center)
                pred_img = pred_img.astype(np.uint8)
            writer.write(pred_img)
        writer.release()
        video_fname = os.path.join(save_path, file_name+'-no_audio.mp4')
        cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(tmp_video_file.name, video_fname)).split()
        call(cmd)
        # add audio
        cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -qscale 0 {2}'.format(wav_file, video_fname, video_fname.replace('-no_audio.mp4', '.mp4'))).split()
        call(cmd)
        if os.path.exists(video_fname):
            os.remove(video_fname)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_file', type=str, default='demos/wav/RD_Radio11_001.wav', help='audio input')
    parser.add_argument('--style_ref', type=str, default='id_002-RD_Radio11_001', help='style reference name')
    parser.add_argument('--checkpoint', type=str, default='pretrained/exp-AR-dim128-IN-style_cls-grl-clip_loss-cycle_style_content-style_SALN-no_sch-epoch_125/Epoch_125.pth', help='checkpoint for inference')
    parser.add_argument('--output_path', type=str, default='demos/results', help='output path')
    parser.add_argument('--use_pytorch3d', type=bool, default=True, help='whether to use PyTorch3D for rendering, if False, pyrender will be used')

    args = parser.parse_args()

    # infer 
    infer = Infer(ckpt=args.checkpoint)
    infer.infer_from_wav(wav_file=args.wav_file, style_ref=args.style_ref, save_path=args.output_path)
