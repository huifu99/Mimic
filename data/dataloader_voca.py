import os
import torch
import numpy as np
import pickle
import re
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data 

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train",read_audio=False,read_text=False):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio
        self.read_text = read_text

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        text_label = self.data[index]['text_label']
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        subject_id = self.data[index]["subject_id"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        if self.read_audio and self.read_text:
            return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), torch.FloatTensor(text_label), subject_id
            # return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), subject_id
        else:
            # return torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name
            return torch.FloatTensor(vertice), torch.FloatTensor(template)

    def __len__(self):
        return self.len

def remove_special_characters(text, chars_to_ignore_regex):
    text_ = re.sub(chars_to_ignore_regex, '', text).upper()
    return text_

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.data_root, args.wav_path)
    text_path = os.path.join(args.data_root, args.text_path)
    vertices_path = os.path.join(args.data_root, args.vertices_path)
    if args.read_audio: # read_audio==False when training vq to save time
        processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2model)

    template_file = os.path.join(args.data_root, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                if args.read_audio:
                    wav_path = os.path.join(r,f)
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values if args.read_audio else None
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]['subject_id'] = subject_id
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:] #due to the memory limit
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)
                
                    sentence_id = int(key.split(".")[0][-2:])
                    txt_path = os.path.join(text_path, subject_id+'.txt')
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                    text = lines[sentence_id-1]
                    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
                    text = remove_special_characters(text, chars_to_ignore_regex)
                    with processor.as_target_processor():
                        text_label = processor(text.strip()).input_ids
                    data[key]["text_label"] = text_label


    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]


    #train vq and pred
    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
    'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}


    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data,subjects_dict,"train",args.read_audio,args.read_text)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_data = Dataset(valid_data,subjects_dict,"val",args.read_audio,args.read_text)
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_data = Dataset(test_data,subjects_dict,"test",args.read_audio,args.read_text)
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return dataset

if __name__ == "__main__":
    get_dataloaders()