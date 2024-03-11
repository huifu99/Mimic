import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

import sys
sys.path.append('.')
from base import BaseModel
from models.lib.base_models import * # Conv1DLN, FeatureProjection, PositionalEncoding, PeriodicPositionalEncoding, init_biased_mask, enc_dec_mask
from models.lib.modules import Mish, get_sinusoid_encoding_table, FFTBlock
from models.lib.wav2vec import Wav2Vec2Model


class DisNetAutoregCycle(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.content_encoder = ContentEncoder(cfg)
        self.style_encoder = StyleEncoder(cfg)
        self.audio_encoder = AudioEncoder(cfg)
        if cfg.style_fuse=='SALN':
            self.decoder = DecoderAutoregSALN(cfg)
        elif cfg.style_fuse=='add':
            self.decoder = DecoderAutoreg(cfg)
        elif cfg.style_fuse=='cat':
            self.decoder = DecoderAutoregCat(cfg)
        elif cfg.style_fuse=='adain':
            self.decoder = DecoderAutoregAdaIN(cfg)
            
        if self.cfg.content_ctc_loss.use:
            self.content_head = ContentClassHead(cfg)
        if self.cfg.content_grl_loss.use:
            self.content_grl_head = GRLClassHead(cfg)
        if self.cfg.content_class_loss.use:
            self.content_cls_head = ContentClassIDHead(cfg)
            self.audio_cls_head = ContentClassIDHead(cfg)
        if self.cfg.style_class_loss.use:
            self.style_class_head = StyleClassHead(cfg)
            if cfg.style_class_loss.use_metrics:
                self.metric_fc = ArcMarginProduct(cfg.feature_dim, cfg.train_ids, s=30, m=0.5, easy_margin=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # tensor(2.6593, grad_fn=<PermuteBackward0>)

    def forward(self, vertices, audio, template, init_state, id_label=None):
        self.device = vertices.device
        motion = vertices-template.unsqueeze(1)
        "content"
        content_code = self.content_encoder(motion)
        content_logits = self.content_head(content_code) if self.cfg.content_ctc_loss.use else None
        content_grl_logits = self.content_grl_head(content_code) if self.cfg.content_grl_loss.use else None
        content_cls_logits = self.content_cls_head(content_code) if self.cfg.content_class_loss.use else None
        "style"
        style_code = self.style_encoder(motion)
        if self.cfg.style_class_loss.use:
            if hasattr(self, 'metric_fc'):
                style_logits = self.metric_fc(style_code, id_label)
            else:
                style_logits = self.style_class_head(style_code)
        else:
            style_logits = None
        "audio"
        audio_feature = self.audio_encoder(audio, frame_num=vertices.shape[1])
        audio_logits = self.content_head(audio_feature) if self.cfg.content_ctc_loss.use else None
        # audio_grl_logits = self.content_grl_head(audio_feature)
        audio_cls_logits = self.audio_cls_head(audio_feature) if self.cfg.content_class_loss.use else None
        
        "decode"
        # audio dec
        dec_out_audio = self.decoder(audio_feature, style_code, init_state)
        dec_out_audio = dec_out_audio+template.unsqueeze(1)
        # content dec
        dec_out_content = self.decoder(content_code, style_code, init_state)
        dec_out_content = dec_out_content+template.unsqueeze(1)
        "cycle"
        batch_size = style_code.size(0)
        style_idx = list(range(batch_size))
        # style_idx_copy = copy.deepcopy(style_idx)
        random.shuffle(style_idx)
        style_code_shuffle = style_code[style_idx, :]
        # decode audio feature
        # dec_out_a_cycle = self.decoder(audio_feature, style_code_shuffle, init_state)
        # style_code_cycle = self.style_encoder(dec_out_a_cycle)
        # content_code_cycle = self.content_encoder(dec_out_a_cycle)
        # decode content code
        dec_out_c_cycle = self.decoder(content_code, style_code_shuffle, init_state)
        style_code_cycle = self.style_encoder(dec_out_c_cycle)
        content_code_cycle = self.content_encoder(dec_out_c_cycle)
        return content_code, style_code, audio_feature, dec_out_content, dec_out_audio, content_logits, audio_logits, content_grl_logits, style_logits, content_cls_logits, audio_cls_logits, style_code_shuffle, style_code_cycle, content_code_cycle, self.logit_scale.exp()

    def predict(self, audio, style_vertices, init_state, template=None, frame_num=None):
        motion = style_vertices-template.unsqueeze(1)
        style_code = self.style_encoder(motion)
        audio_feature = self.audio_encoder(audio, frame_num=frame_num)
        dec_out_audio = self.decoder(audio_feature, style_code, init_state)
        dec_out_audio = dec_out_audio+template.unsqueeze(1)
        return dec_out_audio, audio_feature
    
    def predict_motion(self, audio, style_motion, init_state, frame_num=None):
        style_code = self.style_encoder(style_motion)
        audio_feature = self.audio_encoder(audio, frame_num=frame_num)
        dec_out_audio = self.decoder(audio_feature, style_code, init_state)
        return dec_out_audio, audio_feature

    def predict_motion_content(self, content_motion, style_motion, init_state):
        style_code = self.style_encoder(style_motion)
        content_code = self.content_encoder(content_motion)
        dec_out_content = self.decoder(content_code, style_code, init_state)
        return dec_out_content

    def predict_style_code(self, audio, style_code, init_state, frame_num=None):
        audio_feature = self.audio_encoder(audio, frame_num=frame_num)
        dec_out_audio = self.decoder(audio_feature, style_code, init_state)
        return dec_out_audio


class ContentEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        if cfg.content_norm == 'LN':
            conv_layers = [Conv1DLN(in_conv_dim=cfg.feature_dim, out_conv_dim=cfg.feature_dim, kernel=3, stride=1, padding=1, bias=False) for i in range(cfg.num_conv_layers)]
            if cfg.content_attention:
                self.feature_projection = FeatureProjection(feature_dim=cfg.feature_dim, hidden_size=cfg.hidden_size)
        elif cfg.content_norm == 'IN':
            conv_layers = [Conv1DIN(in_conv_dim=cfg.feature_dim, out_conv_dim=cfg.feature_dim, kernel=3, stride=1, padding=1, bias=False) for i in range(cfg.num_conv_layers)]
            if cfg.content_attention:
                self.feature_projection = FeatureProjectionIN(feature_dim=cfg.feature_dim, hidden_size=cfg.hidden_size)
        self.conv_layers = nn.ModuleList(conv_layers)
        # self.feature_projection = FeatureProjection(feature_dim=cfg.feature_dim, hidden_size=cfg.hidden_size)
        if cfg.content_attention:
            self.PE = PositionalEncoding(cfg.hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_size, nhead=cfg.nhead_encoder, dim_feedforward=cfg.dim_feedforward, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
            self.feature_map = nn.Linear(cfg.hidden_size, cfg.feature_dim)
        else:
            self.feature_map = nn.Linear(cfg.feature_dim, cfg.feature_dim)

    def forward(self, input):
        extract_features = self.motion_map(input)  # (B,L,C)
        extract_features = extract_features.transpose(1,2)  # (B,C,L)
        for conv_layer in self.conv_layers:
            extract_features = conv_layer(extract_features)
        extract_features = extract_features.transpose(1,2)  # (B,L,C)
        if self.cfg.content_attention:
            hidden_states, extract_features = self.feature_projection(extract_features)  # # (B,L,C)
            hidden_states = self.PE(hidden_states)
            hidden_states = self.transformer_encoder(hidden_states)  # (B,L,C)
            content_code = self.feature_map(hidden_states)
        else:
            content_code = self.feature_map(extract_features)
        return content_code


class StyleEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        conv_layers = [Conv1DLN(in_conv_dim=cfg.feature_dim, out_conv_dim=cfg.feature_dim, kernel=3, stride=1, padding=1, bias=False) for i in range(cfg.num_conv_layers)]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.feature_projection = FeatureProjection(feature_dim=cfg.feature_dim, hidden_size=cfg.hidden_size)
        self.PE = PositionalEncoding(cfg.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_size, nhead=cfg.nhead_encoder, dim_feedforward=cfg.dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        self.feature_map = nn.Linear(cfg.hidden_size, cfg.feature_dim)

    def forward(self, input):
        extract_features = self.motion_map(input)  # (B,L,C)
        extract_features = extract_features.transpose(1,2)  # (B,C,L)
        for conv_layer in self.conv_layers:
            extract_features = conv_layer(extract_features)
        extract_features = extract_features.transpose(1,2)  # (B,L,C)
        hidden_states, extract_features = self.feature_projection(extract_features)  # # (B,L,C)
        hidden_states = self.PE(hidden_states)
        hidden_states = self.transformer_encoder(hidden_states)  # (B,L,C)
        style_code = self.feature_map(hidden_states)
        if self.cfg.style_pooling == 'mean':
            style_code = torch.mean(style_code, dim=-2)
        elif self.cfg.style_pooling == 'max':
            style_code = torch.max(style_code, dim=-2)[0]
        return style_code


class StyleEncoderLocal(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        conv_layers = [Conv1DLN(in_conv_dim=cfg.feature_dim, out_conv_dim=cfg.feature_dim, kernel=3, stride=1, padding=1, bias=False) for i in range(cfg.num_conv_layers)]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.feature_projection = FeatureProjection(feature_dim=cfg.feature_dim, hidden_size=cfg.hidden_size)
        self.PE = PositionalEncoding(cfg.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_size, nhead=cfg.nhead_encoder, dim_feedforward=cfg.dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        self.feature_map = nn.Linear(cfg.hidden_size, cfg.feature_dim)

    def forward(self, input):
        extract_features = self.motion_map(input)  # (B,L,C)
        extract_features = extract_features.transpose(1,2)  # (B,C,L)
        for conv_layer in self.conv_layers:
            extract_features = conv_layer(extract_features)
        extract_features = extract_features.transpose(1,2)  # (B,L,C)
        hidden_states, extract_features = self.feature_projection(extract_features)  # # (B,L,C)
        hidden_states = self.PE(hidden_states)
        hidden_states = self.transformer_encoder(hidden_states)  # (B,L,C)
        hidden_states = self.feature_map(hidden_states)
        if self.cfg.style_pooling == 'mean':
            style_code = torch.mean(hidden_states, dim=-2)
        elif self.cfg.style_pooling == 'max':
            style_code = torch.max(hidden_states, dim=-2)[0]
        return style_code, hidden_states


class StyleEncoderLocalV2(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        conv_layers = [
            Conv1DLN(in_conv_dim=cfg.feature_dim, out_conv_dim=cfg.feature_dim, kernel=7, stride=2, padding=3, bias=False),
            Conv1DLN(in_conv_dim=cfg.feature_dim, out_conv_dim=cfg.feature_dim, kernel=5, stride=2, padding=2, bias=False)
        ]
        conv_layers += [Conv1DLN(in_conv_dim=cfg.feature_dim, out_conv_dim=cfg.feature_dim, kernel=3, stride=1, padding=1, bias=False) for i in range(cfg.num_conv_layers-2)]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.feature_projection = FeatureProjection(feature_dim=cfg.feature_dim, hidden_size=cfg.hidden_size)
        self.PE = PositionalEncoding(cfg.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_size, nhead=cfg.nhead_encoder, dim_feedforward=cfg.dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        self.feature_map = nn.Linear(cfg.hidden_size, cfg.feature_dim)

    def forward(self, input):
        extract_features = self.motion_map(input)  # (B,L,C)
        extract_features = extract_features.transpose(1,2)  # (B,C,L)
        for conv_layer in self.conv_layers:
            extract_features = conv_layer(extract_features)
        extract_features = extract_features.transpose(1,2)  # (B,L,C)
        hidden_states, extract_features = self.feature_projection(extract_features)  # # (B,L,C)
        hidden_states = self.PE(hidden_states)
        hidden_states = self.transformer_encoder(hidden_states)  # (B,L,C)
        hidden_states = self.feature_map(hidden_states)
        if self.cfg.style_pooling == 'mean':
            style_code = torch.mean(hidden_states, dim=-2)
        elif self.cfg.style_pooling == 'max':
            style_code = torch.max(hidden_states, dim=-2)[0]
        return style_code, hidden_states


class StyleEncoderV2(nn.Module):  # need to debug
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        conv_layers = [Conv1DLN(in_conv_dim=cfg.feature_dim, out_conv_dim=cfg.feature_dim, kernel=3, stride=2, padding=1, bias=False) for i in range(cfg.num_conv_layers)]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.feature_projection = FeatureProjection(feature_dim=cfg.feature_dim, hidden_size=cfg.hidden_size)
        self.PE = PositionalEncoding(cfg.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_size, nhead=cfg.nhead_encoder, dim_feedforward=cfg.dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        self.feature_map = nn.Linear(cfg.hidden_size, cfg.feature_dim)

    def forward(self, input):
        extract_features = self.motion_map(input)  # (B,L,C)
        extract_features = extract_features.transpose(1,2)  # (B,C,L)
        for conv_layer in self.conv_layers:
            extract_features = conv_layer(extract_features)
        extract_features = extract_features.transpose(1,2)  # (B,L,C)
        hidden_states, extract_features = self.feature_projection(extract_features)  # # (B,L,C)
        hidden_states = self.PE(hidden_states)
        hidden_states = self.transformer_encoder(hidden_states)  # (B,L,C)
        style_code = self.feature_map(hidden_states)
        if self.cfg.style_pooling == 'mean':
            style_code = torch.mean(style_code, dim=-2)
        elif self.cfg.style_pooling == 'max':
            style_code = torch.max(style_code, dim=-2)[0]
        return style_code

class AudioEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset = cfg.dataset
        self.audio_encoder = Wav2Vec2Model.from_pretrained(cfg.wav2vec2model)
        if cfg.freeze_TCN:
            self.audio_encoder.feature_extractor._freeze_parameters()
        if cfg.audio_feature_align == 'conv':
            self.down_sample = torch.nn.Conv1d(cfg.audio_hidden_size, cfg.audio_hidden_size, kernel_size=15, stride=2, padding=7)
        self.audio_feature_map = nn.Linear(cfg.audio_hidden_size, cfg.feature_dim)
    
    def forward(self, audio, frame_num=None):
        if self.cfg.freeze_audio_encoder:
            with torch.no_grad():
                hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num, align=self.cfg.audio_feature_align).last_hidden_state
        else:
            hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num, align=self.cfg.audio_feature_align).last_hidden_state
        if hasattr(self, 'down_sample'):
            # hidden_states = hidden_states.repeat_interleave(self.ur, 1)
            hidden_states = self.down_sample(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.audio_feature_map(hidden_states)
        return hidden_states


class DecoderAutoreg(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.PPE = PeriodicPositionalEncoding(cfg.feature_dim, period = 30)
        self.biased_mask = init_biased_mask(n_head = cfg.decoder.nhead, max_seq_len = 600, period=30)
        decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.feature_dim, nhead=cfg.decoder.nhead, dim_feedforward=cfg.decoder.dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder.num_layers)
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        self.motion_map_r = nn.Linear(cfg.feature_dim, cfg.motion_dim)
        nn.init.constant_(self.motion_map_r.weight, 0)
        nn.init.constant_(self.motion_map_r.bias, 0)

    def forward(self, content_code, style_code, init_state):
        self.device = content_code.device
        # auto reg
        frame_num = content_code.shape[1]
        for i in range(frame_num):
            if i == 0:
                dec_emb = self.motion_map(init_state).unsqueeze(1)
                style_emb = style_code.unsqueeze(1)
                dec_emb = dec_emb+style_emb
                # dec_input = self.PPE(style_emb)
                dec_input = self.PPE(dec_emb)
            else:
                dec_input = self.PPE(dec_emb)
            tgt_mask = self.biased_mask[:, :dec_input.shape[1], :dec_input.shape[1]].clone().detach().to(device=content_code.device).repeat(dec_input.shape[0], 1, 1)
            memory_mask = enc_dec_mask(self.device, dec_input.shape[1], content_code.shape[1])
            dec_out = self.transformer_decoder(dec_input, content_code, tgt_mask=tgt_mask, memory_mask=memory_mask)
            dec_out = self.motion_map_r(dec_out)
            new_out = self.motion_map(dec_out[:,-1,:]).unsqueeze(1)
            new_out = new_out+style_emb
            dec_emb = torch.cat((dec_emb, new_out), dim=1)
        return dec_out


class DecoderAutoregCat(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.PPE = PeriodicPositionalEncoding(cfg.feature_dim, period = 30)
        self.biased_mask = init_biased_mask(n_head = cfg.decoder.nhead, max_seq_len = 600, period=30)
        decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.feature_dim, nhead=cfg.decoder.nhead, dim_feedforward=cfg.decoder.dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder.num_layers)
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        self.motion_map_r = nn.Linear(cfg.feature_dim, cfg.motion_dim)
        nn.init.constant_(self.motion_map_r.weight, 0)
        nn.init.constant_(self.motion_map_r.bias, 0)
        self.style_map = nn.Linear(cfg.feature_dim*2, cfg.feature_dim)

    def forward(self, content_code, style_code, init_state):
        self.device = content_code.device
        # auto reg
        frame_num = content_code.shape[1]
        for i in range(frame_num):
            if i == 0:
                dec_emb = self.motion_map(init_state).unsqueeze(1)
                style_emb = style_code.unsqueeze(1)
                # dec_emb = dec_emb+style_emb
                dec_emb = self.style_map(torch.cat([dec_emb, style_emb], dim=-1))
                # dec_input = self.PPE(style_emb)
                dec_input = self.PPE(dec_emb)
            else:
                dec_input = self.PPE(dec_emb)
            tgt_mask = self.biased_mask[:, :dec_input.shape[1], :dec_input.shape[1]].clone().detach().to(device=content_code.device).repeat(dec_input.shape[0], 1, 1)
            memory_mask = enc_dec_mask(self.device, dec_input.shape[1], content_code.shape[1])
            dec_out = self.transformer_decoder(dec_input, content_code, tgt_mask=tgt_mask, memory_mask=memory_mask)
            dec_out = self.motion_map_r(dec_out)
            new_out = self.motion_map(dec_out[:,-1,:]).unsqueeze(1)
            # new_out = new_out+style_emb
            new_out = self.style_map(torch.cat([new_out, style_emb], dim=-1))
            dec_emb = torch.cat((dec_emb, new_out), dim=1)
        return dec_out


class DecoderAutoregAdaIN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.PPE = PeriodicPositionalEncoding(cfg.feature_dim, period = 30)
        self.biased_mask = init_biased_mask(n_head = cfg.decoder.nhead, max_seq_len = 600, period=30)
        decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.feature_dim, nhead=cfg.decoder.nhead, dim_feedforward=cfg.decoder.dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder.num_layers)
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        self.motion_map_r = nn.Linear(cfg.feature_dim, cfg.motion_dim)
        nn.init.constant_(self.motion_map_r.weight, 0)
        nn.init.constant_(self.motion_map_r.bias, 0)
        # self.style_map = nn.Linear(cfg.feature_dim*2, cfg.feature_dim)
        self.adain = AdaIN(cfg.feature_dim, cfg.feature_dim)

    def forward(self, content_code, style_code, init_state):
        self.device = content_code.device
        # auto reg
        frame_num = content_code.shape[1]
        content_code = self.adain(content_code, style_code)
        for i in range(frame_num):
            if i == 0:
                dec_emb = self.motion_map(init_state).unsqueeze(1)
                # style_emb = style_code.unsqueeze(1)
                # dec_emb = dec_emb+style_emb
                # dec_emb = self.style_map(torch.cat([dec_emb, style_emb], dim=-1))
                # dec_emb = self.adain(dec_emb, style_code)
                # dec_input = self.PPE(style_emb)
                dec_input = self.PPE(dec_emb)
            else:
                dec_input = self.PPE(dec_emb)
            tgt_mask = self.biased_mask[:, :dec_input.shape[1], :dec_input.shape[1]].clone().detach().to(device=content_code.device).repeat(dec_input.shape[0], 1, 1)
            memory_mask = enc_dec_mask(self.device, dec_input.shape[1], content_code.shape[1])
            dec_out = self.transformer_decoder(dec_input, content_code, tgt_mask=tgt_mask, memory_mask=memory_mask)
            dec_out = self.motion_map_r(dec_out)
            new_out = self.motion_map(dec_out[:,-1,:]).unsqueeze(1)
            # new_out = new_out+style_emb
            # new_out = self.style_map(torch.cat([new_out, style_emb], dim=-1))
            # new_out = self.adain(new_out, style_code)
            dec_emb = torch.cat((dec_emb, new_out), dim=1)
        return dec_out


class DecoderAutoregSALN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.PPE = PeriodicPositionalEncoding(cfg.feature_dim, period = 30)
        self.biased_mask = init_biased_mask(n_head = cfg.decoder.nhead, max_seq_len = 600, period=30)
        decoder_layer = TransformerDecoderLayer(d_model=cfg.feature_dim, nhead=cfg.decoder.nhead, dim_feedforward=cfg.decoder.dim_feedforward, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=cfg.decoder.num_layers)
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        self.motion_map_r = nn.Linear(cfg.feature_dim, cfg.motion_dim)
        nn.init.constant_(self.motion_map_r.weight, 0)
        nn.init.constant_(self.motion_map_r.bias, 0)

    def forward(self, content_code, style_code, init_state):
        self.device = content_code.device
        # auto reg
        frame_num = content_code.shape[1]
        for i in range(frame_num):
            if i == 0:
                dec_emb = self.motion_map(init_state).unsqueeze(1)
                # style_emb = style_code.unsqueeze(1)
                # dec_emb = dec_emb +style_emb
                dec_input = self.PPE(dec_emb)
            else:
                dec_input = self.PPE(dec_emb)
            tgt_mask = self.biased_mask[:, :dec_input.shape[1], :dec_input.shape[1]].clone().detach().to(device=content_code.device).repeat(dec_input.shape[0], 1, 1)
            memory_mask = enc_dec_mask(self.device, dec_input.shape[1], content_code.shape[1])
            dec_out = self.transformer_decoder(dec_input, content_code, tgt_mask=tgt_mask, memory_mask=memory_mask, style_code=style_code)
            dec_out = self.motion_map_r(dec_out)
            new_out = self.motion_map(dec_out[:,-1,:]).unsqueeze(1)
            # new_out = new_out+style_emb
            dec_emb = torch.cat((dec_emb, new_out), dim=1)
        return dec_out
    

class DecoderAutoregCrossAtt(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cross_att = nn.MultiheadAttention(embed_dim=cfg.feature_dim, num_heads=1, batch_first=True)
        self.style_emb_map = nn.Linear(cfg.feature_dim*2, cfg.feature_dim)
        self.PPE = PeriodicPositionalEncoding(cfg.feature_dim, period = 30)
        self.biased_mask = init_biased_mask(n_head = cfg.decoder.nhead, max_seq_len = 600, period=30)
        decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.feature_dim, nhead=cfg.decoder.nhead, dim_feedforward=cfg.decoder.dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder.num_layers)
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        self.motion_map_r = nn.Linear(cfg.feature_dim, cfg.motion_dim)
        nn.init.constant_(self.motion_map_r.weight, 0)
        nn.init.constant_(self.motion_map_r.bias, 0)

    def forward(self, content_code, style_code, style_hiddens, init_state):
        self.device = content_code.device
        style_emb_g = style_code.unsqueeze(1)
        style_emb_l_list = []
        # auto reg
        frame_num = content_code.shape[1]
        for i in range(frame_num):
            if i == 0:
                dec_emb = self.motion_map(init_state).unsqueeze(1)
                # local style_emb
                q, k, v = dec_emb, style_hiddens, style_hiddens
                style_emb_l, style_emb_l_w = self.cross_att(q, k, v)
                # add
                # dec_emb = dec_emb+style_emb_g+style_emb_l
                # cat_add
                dec_emb = self.style_emb_map(torch.cat([dec_emb, style_emb_l], dim=-1))+style_emb_g
                dec_input = self.PPE(dec_emb)
            else:
                dec_input = self.PPE(dec_emb)
            tgt_mask = self.biased_mask[:, :dec_input.shape[1], :dec_input.shape[1]].clone().detach().to(device=content_code.device).repeat(dec_input.shape[0], 1, 1)
            memory_mask = enc_dec_mask(self.device, dec_input.shape[1], content_code.shape[1])
            dec_out = self.transformer_decoder(dec_input, content_code, tgt_mask=tgt_mask, memory_mask=memory_mask)
            dec_out = self.motion_map_r(dec_out)
            new_out = self.motion_map(dec_out[:,-1,:]).unsqueeze(1)
            q = new_out
            style_emb_l, style_emb_l_w = self.cross_att(q, k, v)
            # add
            # new_out = new_out+style_emb_g+style_emb_l
            # cat_add
            new_out = self.style_emb_map(torch.cat([new_out, style_emb_l], dim=-1))+style_emb_g
            style_emb_l_list.append(style_emb_l)
            dec_emb = torch.cat((dec_emb, new_out), dim=1)
        return dec_out, torch.cat(style_emb_l_list, dim=1)


class DecoderAutoregCrossAttV2(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cross_att = nn.MultiheadAttention(embed_dim=cfg.feature_dim, num_heads=1, batch_first=True)
        self.style_emb_map = nn.Linear(cfg.feature_dim*2, cfg.feature_dim)
        self.PPE = PeriodicPositionalEncoding(cfg.feature_dim, period = 30)
        self.biased_mask = init_biased_mask(n_head = cfg.decoder.nhead, max_seq_len = 600, period=30)
        decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.feature_dim, nhead=cfg.decoder.nhead, dim_feedforward=cfg.decoder.dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder.num_layers)
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        self.motion_map_r = nn.Linear(cfg.feature_dim, cfg.motion_dim)
        nn.init.constant_(self.motion_map_r.weight, 0)
        nn.init.constant_(self.motion_map_r.bias, 0)

    def forward(self, content_code, style_code, style_hiddens, init_state):
        self.device = content_code.device
        style_emb_g = style_code.unsqueeze(1)
        k, v = style_hiddens, style_hiddens
        style_emb_l_list = []
        # auto reg
        frame_num = content_code.shape[1]
        for i in range(frame_num):
            if i == 0:
                dec_emb = self.motion_map(init_state).unsqueeze(1)
                # local style_emb
                q = content_code[:,i:i+1,:]
                style_emb_l, style_emb_l_w = self.cross_att(q, k, v)
                # add
                # dec_emb = dec_emb+style_emb_g+style_emb_l
                # cat_add
                dec_emb = self.style_emb_map(torch.cat([dec_emb, style_emb_l], dim=-1))+style_emb_g
                dec_input = self.PPE(dec_emb)
            else:
                dec_input = self.PPE(dec_emb)
            tgt_mask = self.biased_mask[:, :dec_input.shape[1], :dec_input.shape[1]].clone().detach().to(device=content_code.device).repeat(dec_input.shape[0], 1, 1)
            memory_mask = enc_dec_mask(self.device, dec_input.shape[1], content_code.shape[1])
            dec_out = self.transformer_decoder(dec_input, content_code, tgt_mask=tgt_mask, memory_mask=memory_mask)
            dec_out = self.motion_map_r(dec_out)
            new_out = self.motion_map(dec_out[:,-1,:]).unsqueeze(1)
            q = content_code[:,i:i+1,:]
            style_emb_l, style_emb_l_w = self.cross_att(q, k, v)
            # add
            # new_out = new_out+style_emb_g+style_emb_l
            # cat_add
            new_out = self.style_emb_map(torch.cat([new_out, style_emb_l], dim=-1))+style_emb_g
            style_emb_l_list.append(style_emb_l)
            dec_emb = torch.cat((dec_emb, new_out), dim=1)
        return dec_out, torch.cat(style_emb_l_list, dim=1)


class DecoderAutoregCrossAttV3(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.style_emb_map = nn.Linear(cfg.feature_dim*2, cfg.feature_dim)
        self.PPE = PeriodicPositionalEncoding(cfg.feature_dim, period = 30)
        self.biased_mask = init_biased_mask(n_head = cfg.decoder.nhead, max_seq_len = 600, period=30)
        decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.feature_dim, nhead=cfg.decoder.nhead, dim_feedforward=cfg.decoder.dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.decoder.num_layers)
        self.motion_map = nn.Linear(cfg.motion_dim, cfg.feature_dim)
        self.motion_map_r = nn.Linear(cfg.feature_dim, cfg.motion_dim)
        nn.init.constant_(self.motion_map_r.weight, 0)
        nn.init.constant_(self.motion_map_r.bias, 0)

    def forward(self, content_code, style_code, style_code_l, init_state):
        self.device = content_code.device
        style_emb_g = style_code.unsqueeze(1)
        # auto reg
        frame_num = content_code.shape[1]
        for i in range(frame_num):
            if i == 0:
                dec_emb = self.motion_map(init_state).unsqueeze(1)
                # add
                dec_emb = dec_emb+style_emb_g+style_code_l[:,i:i+1,:]
                # cat_add
                # dec_emb = self.style_emb_map(torch.cat([dec_emb, style_emb_l], dim=-1))+style_emb_g
                dec_input = self.PPE(dec_emb)
            else:
                dec_input = self.PPE(dec_emb)
            tgt_mask = self.biased_mask[:, :dec_input.shape[1], :dec_input.shape[1]].clone().detach().to(device=content_code.device).repeat(dec_input.shape[0], 1, 1)
            memory_mask = enc_dec_mask(self.device, dec_input.shape[1], content_code.shape[1])
            dec_out = self.transformer_decoder(dec_input, content_code, tgt_mask=tgt_mask, memory_mask=memory_mask)
            dec_out = self.motion_map_r(dec_out)
            new_out = self.motion_map(dec_out[:,-1,:]).unsqueeze(1)
            # add
            new_out = new_out+style_emb_g+style_code_l[:,i:i+1,:]
            # cat_add
            # new_out = self.style_emb_map(torch.cat([new_out, style_emb_l], dim=-1))+style_emb_g
            dec_emb = torch.cat((dec_emb, new_out), dim=1)
        return dec_out


class Decoder(nn.Module):
    """ Non-AutoReg Decoder """
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        config = cfg.decoder_NAR
        self.max_seq_len = config.max_seq_len
        self.n_layers = config.decoder_layers
        self.d_model = config.decoder_hidden
        self.n_head = config.decoder_head
        self.d_k = config.decoder_hidden // config.decoder_head
        self.d_v = config.decoder_hidden // config.decoder_head
        self.d_inner = config.fft_conv1d_filter_size
        self.fft_conv1d_kernel_size = config.fft_conv1d_kernel_size
        self.d_out = cfg.motion_dim
        self.style_dim = cfg.feature_dim
        self.dropout = 0.1

        self.prenet = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model//2, self.d_model)
        )

        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, 
            self.fft_conv1d_kernel_size, self.style_dim, self.dropout) for _ in range(self.n_layers)])

        self.fc_out = nn.Linear(self.d_model, self.d_out)

    def forward(self, enc_seq, style_code, mask=None):
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        # -- Prepare masks
        mask = torch.zeros(batch_size, max_len).to(enc_seq.device).bool()  # no mask
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # prenet
        dec_embedded = self.prenet(enc_seq)
        # poistion encoding
        if enc_seq.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model)[:enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(enc_seq.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        dec_output = dec_embedded + position_embedded
        # fft blocks
        # slf_attn = []
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, style_code,
                mask=mask,
                slf_attn_mask=slf_attn_mask)
            # slf_attn.append(dec_slf_attn)
        # last fc
        dec_output = self.fc_out(dec_output)
        return dec_output # , slf_attn



if __name__ == '__main__':
    from base.utilities import get_parser
    cfg = get_parser()
    "test encoder"
    # input = torch.randn((4, 150, 15069)).cuda()
    # content_encoder = ContentEncoder(cfg).cuda()
    # content_code = content_encoder(input)

    "test decoder"
    content_code = torch.randn((4, 150, 512)).cuda()
    style_code = torch.randn((4, 512)).cuda()
    decoder = Decoder(cfg).cuda()
    output = decoder(content_code, style_code)