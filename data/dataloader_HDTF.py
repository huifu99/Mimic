import os
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import numpy as np
import pickle
import copy
import random
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, cfg, data_type="train") -> None:
        super().__init__()
        self.cfg = cfg
        self.data_type = data_type
        self.fps = cfg.video_fps
        self.sentence_num = cfg.sentence_num
        self.window = cfg.window
        self.clip_overlap = self.cfg.clip_overlap
        self.data_root = cfg.data_root
        self.audio_path = os.path.join(self.data_root, cfg.audio_path)
        self.codedict_path = os.path.join(self.data_root, cfg.codedict_path)
        self.video_list_file = os.path.join(self.data_root, cfg.video_list_file)
        template_file = os.path.join(self.data_root, cfg.template_file)
        # with open(template_file, 'rb') as fin:
        self.templates = pickle.load(open(template_file, 'rb'))

        self.processor = Wav2Vec2Processor.from_pretrained(cfg.wav2vec2model)
        
        self.get_data()
        self.one_hot_init = np.eye(cfg.train_ids)
        if data_type == 'train':
            train_id_list = list(range(cfg.train_ids))
            id_list = [id for id in self.id_list if id-1 in train_id_list]
            video_list = []
            for i, id in enumerate(self.id_list):
                if id-1 in train_id_list:
                    video_list.append(self.video_list[i])
            self.id_list = id_list
            self.video_list = video_list
            
        self.id_set = sorted(list(set(self.id_list)))
        print('train_ids: ', self.id_set)

    def __len__(self):
        return len(self.video_list)*self.sentence_num

    def __getitem__(self, index):
        video_idx = index%len(self.video_list)
        video_label = self.video_list[video_idx]
        # subject_id = video_label.split('_')[1]
        subject_id = self.id_list[video_idx]
        # onehot
        # one_hot = self.one_hot_init[self.id_list.index(subject_id)]
        one_hot = self.one_hot_init[subject_id-1]

        # template
        template = self.templates[subject_id].reshape(-1)
        # vertices
        codedict_npy = os.path.join(self.codedict_path, video_label, 'verts_new_shape1.npy')
        codedict = np.load(codedict_npy, allow_pickle=True).item()
        vertices_all = codedict['verts'].reshape(-1, self.cfg.motion_dim)
        frame_num = vertices_all.shape[0]
        if self.clip_overlap:
            max_frame = frame_num-self.window-1
            start_frame = random.randint(0, max_frame)
        else:
            max_idx = frame_num//self.window-1
            start_idx = random.randint(0, max_idx)
            start_frame = start_idx*self.window
        vertices = vertices_all[start_frame: start_frame+self.window, :]
        # init_state
        init_state = np.zeros((vertices.shape[-1])) if start_frame==0 else vertices_all[start_frame-1, :]

        # audio
        # huggingface
        # wav_path = os.path.join(self.audio_path, video_label+'.wav')
        # speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        audio_npy = os.path.join(self.codedict_path, video_label, 'audio_librosa.npy')
        audio_read = np.load(audio_npy, allow_pickle=True).item()
        speech_array, sampling_rate = audio_read['speech_array'], audio_read['sampling_rate']
        stride = int(sampling_rate//self.fps)
        audio = speech_array[start_frame*stride: (start_frame+self.window)*stride]
        audio = np.squeeze(self.processor(audio,sampling_rate=16000).input_values)
        if audio.shape[0]<96000:
            audio = np.concatenate([audio, np.zeros((96000-audio.shape[0]))], axis=0)

        return {
            'audio': torch.FloatTensor(audio),
            'vertices': torch.FloatTensor(vertices),
            'template': torch.FloatTensor(template),
            'one_hot': torch.FloatTensor(one_hot),
            'subject_id': subject_id,
            'init_state': torch.FloatTensor(init_state),
        }


    def get_data(self):
        self.video_dict = {}
        with open(self.video_list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if self.data_type=='train' and line.strip().split(' ')[-1] != 'test':
                    video = line.strip().split(' ')[0]
                    id = int(line.strip().split(' ')[1])
                    self.video_dict[video] = id
        self.video_list = list(self.video_dict.keys())
        self.id_list = list(self.video_dict.values())


def get_dataloaders(cfg):
    dataset = {}
    train_data = Dataset(cfg,data_type="train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    # valid_data = Dataset(cfg,data_type="val")
    # dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    return dataset


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from base.utilities import get_parser
    cfg, params_dict = get_parser()
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']
    for i, d in enumerate(train_loader):
        print(d)