from einops.layers.torch import Rearrange
import copy
import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
from typing import Optional, Any, Union, Callable

from utils.base_model_util import *
from models.lib.grl_module import GradientReversal
from models.lib.modules import StyleAdaptiveLayerNorm

class Norm(nn.Module):
  """ Norm Layer """

  def __init__(self, fn, size):
    super().__init__()
    self.norm = nn.LayerNorm(size, eps=1e-5)
    self.fn = fn

  def forward(self, x_data):
    if type(x_data) is dict:
        x_norm = self.fn({'x_a':x_data['x_a'], 'x_b':self.norm(x_data['x_b'])})
        return x_norm
    else:
        x, mask_info = x_data
        x_norm, _ = self.fn((self.norm(x), mask_info))
        return (x_norm, mask_info)

class Residual(nn.Module):
  """ Residual Layer """

  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x_data):
    if type(x_data) is dict:
        x_resid = self.fn(x_data)['x_b']
        return {'x_a':x_data['x_a'], 'x_b':x_resid+x_data['x_b']}
    else:
        x, mask_info = x_data
        x_resid, _ = self.fn(x_data)
        return (x_resid + x, mask_info)


class MLP(nn.Module):
  """ MLP Layer """

  def __init__(self, in_dim, out_dim, hidden_dim):
    super().__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.activation = get_activation("gelu")
    self.l2 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x_data):
    if type(x_data) is dict:
        out = self.l2(self.activation(self.l1(x_data['x_b'])))
        return {'x_a':x_data['x_a'], 'x_b':out}
    else:
        x, mask_info = x_data
        out = self.l2(self.activation(self.l1(x)))
        return (out, mask_info)


class CrossModalAttention(nn.Module):
  """ Cross Modal Attention Layer
  Given 2 modalities (a, b), computes the K,V from modality b and Q from
  modality a.
  """

  def __init__(self, in_dim, dim, heads=8, in_dim2=None):
    super().__init__()
    self.heads = heads
    self.scale = dim**-0.5

    if in_dim2 is not None:
        self.to_kv = nn.Linear(in_dim2, in_dim2 * 2, bias=False)
    else:
        self.to_kv = nn.Linear(in_dim, dim * 2, bias=False)
    self.to_q = nn.Linear(in_dim, dim, bias=False)
    if in_dim2 is not None:
        dim2 = int((in_dim + in_dim2*2) / 3)
    else:
        dim2 = dim
    self.to_out = nn.Linear(dim2, dim)

    self.rearrange_qkv = Rearrange(
        "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
    self.rearrange_out = Rearrange("b h n d -> b n (h d)")

  def forward(self, x_data):
    x_a = x_data['x_a']
    x_b = x_data['x_b']

    kv = self.to_kv(x_b)
    q = self.to_q(x_a)

    qkv = torch.cat((q, kv), dim=-1)
    qkv = self.rearrange_qkv(qkv)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
    attn = F.softmax(dots, dim=-1)

    out = torch.einsum("bhij,bhjd->bhid", attn, v)
    out = self.rearrange_out(out)
    out = self.to_out(out)
    return {'x_a':x_a, 'x_b':out}


class Attention(nn.Module):
  """ Attention Layer """

  def __init__(self, in_dim, dim, heads=8):
    super().__init__()
    self.heads = heads
    self.scale = dim**-0.5

    self.to_qkv = nn.Linear(in_dim, dim * 3, bias=False)
    self.to_out = nn.Linear(dim, dim)

    self.rearrange_qkv = Rearrange(
        "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
    self.rearrange_out = Rearrange("b h n d -> b n (h d)")

  def forward(self, x_data):
    x, mask_info = x_data
    max_mask = mask_info['max_mask']
    mask = mask_info['mask']
    #
    qkv = self.to_qkv(x)
    qkv = self.rearrange_qkv(qkv)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
    if max_mask is not None:
        dots[:,:,:max_mask,:max_mask] = \
            dots[:,:,:max_mask,:max_mask].masked_fill(mask == 0., float('-inf'))

    attn = F.softmax(dots, dim=-1)

    out = torch.einsum("bhij,bhjd->bhid", attn, v)
    out = self.rearrange_out(out)
    out = self.to_out(out)
    return (out, mask_info)


class Transformer(nn.Module):
  """ Transformer class
  Parameters
  ----------
  cross_modal : bool
    if true, uses cross-modal attention layers, else is the vanilla Transformer
  in_dim2 : int
    specifies the feature size of the second modality if using cross_modal
  """

  def __init__(self,
               in_size=50,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               cross_modal=False,
               in_dim2=None):
    super().__init__()
    blocks = []
    attn = False

    self.cross_modal = cross_modal
    if cross_modal:
      for i in range(num_hidden_layers):
        blocks.extend([
            Residual(Norm(CrossModalAttention(in_size, hidden_size,
                                              heads=num_attention_heads,
                                              in_dim2=in_dim2), hidden_size)),
            Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size),
                              hidden_size))
        ])
    else:
      for i in range(num_hidden_layers):
        blocks.extend([
            Residual(Norm(Attention(in_size, hidden_size,
                                    heads=num_attention_heads), hidden_size)),
            Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size),
                              hidden_size))
        ])
    self.net = torch.nn.Sequential(*blocks)

  def forward(self, x_data):
    if self.cross_modal:
      assert type(x_data) is dict
      x_data = self.net(x_data)
      x = x_data['x_b']
    else:
      x, mask_info = x_data
      x, _ = self.net((x, mask_info))
    return x


class LinearEmbedding(nn.Module):
  """ Linear Layer """

  def __init__(self, size, dim):
    super().__init__()
    self.net = nn.Linear(size, dim)

  def forward(self, x):
    return self.net(x)


class AudioEmbedding(nn.Module):
  """ Audio embedding layer
  Parameters
  ----------
  size : int
    the input feature size of the audio embedding
  dim : int
    the desired output feature size for the audio embedding
  quant_factor: int
    specifies the number of max pool layers applied along the temporal dimension
  version: str (default is 'v6')
    specifies which version of the audio embedding to use
  """

  def __init__(self, size, dim, quant_factor, version='v6'):
    super().__init__()
    self.proj = None
    if version == 'v6':
        print('MODEL V6')
        self.net = nn.MaxPool1d(4)
        layers = [nn.Sequential(nn.MaxPool1d(2))]
        for _ in range(1, quant_factor):
            layers += [nn.Sequential(
                           nn.MaxPool1d(2)
                           )]
        self.squasher = nn.Sequential(*layers)
        self.proj = nn.Linear(size,dim)

  def forward(self, x):
    x = self.net(x)
    x = self.squasher(x)
    if self.proj is not None:
        x = self.proj(x.permute(0,2,1)).permute(0,2,1)
    return x

class PositionEmbedding(nn.Module):
  """Postion Embedding Layer"""

  def __init__(self, seq_length, dim):
    super().__init__()
    self.pos_embedding = nn.Parameter(torch.zeros(seq_length, dim))

  def forward(self, x):
    return x + self.pos_embedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CrossModalLayer(nn.Module):
  """Cross Modal Layer inspired by FACT [Li 2021]"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    model_config = self.config['transformer']
    self.transformer_layer = Transformer(
        in_size=model_config['hidden_size'],
        hidden_size=model_config['hidden_size'],
        num_hidden_layers=model_config['num_hidden_layers'],
        num_attention_heads=model_config['num_attention_heads'],
        intermediate_size=model_config['intermediate_size'])

    output_layer_config = self.config['output_layer']
    self.cross_norm_layer = nn.LayerNorm(self.config['in_dim'])
    self.cross_output_layer = nn.Linear(
                                    self.config['in_dim'],
                                    output_layer_config['out_dim'],
                                    bias=False)

    self.cross_pos_embedding = PositionEmbedding(
            self.config["sequence_length"], self.config['in_dim'])


  def forward(self, modal_a_sequences, modal_b_sequences, mask_info):
    """
    Parameters
    ----------
    modal_a_sequences : tensor
        the first modality (e.g. Listener motion embedding)
    modal_b_sequences : tensor
        the second modality (e.g. Speaker motion+audio embedding)
    mask_info: dict
        specifies the binary mask that is applied to the Transformer attention
    """

    _, _, modal_a_width = get_shape_list(modal_a_sequences)
    merged_sequences = modal_a_sequences
    if modal_b_sequences is not None:
        _, _, modal_b_width = get_shape_list(modal_b_sequences)
        if modal_a_width != modal_b_width:
          raise ValueError(
              "The modal_a hidden size (%d) should be the same with the modal_b"
              "hidden size (%d)" % (modal_a_width, modal_b_width))
        merged_sequences = torch.cat([merged_sequences, modal_b_sequences],
                                      axis=1)

    merged_sequences = self.cross_pos_embedding(merged_sequences)
    merged_sequences = self.transformer_layer((merged_sequences, mask_info))
    merged_sequences = self.cross_norm_layer(merged_sequences)
    logits = self.cross_output_layer(merged_sequences)
    return logits


class Conv1DLN(nn.Module):
    def __init__(self, in_conv_dim, out_conv_dim, kernel, stride, padding, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.layer_norm = nn.LayerNorm(out_conv_dim, elementwise_affine=True)
        self.activation = F.relu

    def forward(self, hidden_states):  # hidden_states: (B,C,L)
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)  # (B,L,C)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)  # (B,C,L)

        hidden_states = self.activation(hidden_states, inplace=True)
        return hidden_states

class Conv1DIN(nn.Module):
    def __init__(self, in_conv_dim, out_conv_dim, kernel, stride, padding, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.IN = nn.InstanceNorm1d(out_conv_dim, affine=False)
        self.activation = F.relu

    def forward(self, hidden_states):  # hidden_states: (B,C,L)
        hidden_states = self.conv(hidden_states)

        # hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.IN(hidden_states)
        # hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class FeatureProjection(nn.Module):
    def __init__(self, feature_dim, hidden_size, feat_proj_dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_size)
        self.dropout = nn.Dropout(feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class FeatureProjectionIN(nn.Module):
    def __init__(self, feature_dim, hidden_size, feat_proj_dropout=0.0):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_size)
        self.dropout = nn.Dropout(feat_proj_dropout)

    def forward(self, hidden_states):  # hidden_states (B,L,C)
        # non-projected hidden states are needed for quantization
        hidden_states = hidden_states.transpose(-2,-1)  # (B,C,L)
        norm_hidden_states = self.instance_norm(hidden_states)
        norm_hidden_states = norm_hidden_states.transpose(-2,-1)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Alignment Bias
def enc_dec_mask(device, T, S, dataset='vocaset'):
    mask = torch.ones(T, S)  # (T,frame_num)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

class ContentClassHead(nn.Module):
  def __init__(self, cfg) -> None:
    super().__init__()
    self.lm_head = nn.Linear(cfg.feature_dim, cfg.vocab_size)
    
  def forward(self, feature):
    logits = self.lm_head(feature)
    return logits

class ContentClassIDHead(nn.Module):
  def __init__(self, cfg) -> None:
    super().__init__()
    id_num = cfg.train_ids
    self.class_head = nn.Linear(cfg.feature_dim, id_num)
    
  def forward(self, feature):
    feature = torch.mean(feature, dim=-2)
    logits = self.class_head(feature)
    return logits

class GRLClassHead(nn.Module):
  def __init__(self, cfg) -> None:
    super().__init__()
    self.cfg = cfg
    id_num = cfg.train_ids
    self.grl = GradientReversal(alpha=cfg.content_grl_loss.alpha)
    self.class_head = nn.Linear(cfg.feature_dim, id_num)
    
  def forward(self, feature):
    feature_ = torch.mean(feature, dim=-2)
    if self.cfg.content_grl_loss.use_grl:
      feature_ = self.grl(feature_)
    logits = self.class_head(feature_)
    return logits
  

class StyleClassHead(nn.Module):
  def __init__(self, cfg) -> None:
    super().__init__()
    id_num = cfg.train_ids
    self.class_head = nn.Linear(cfg.feature_dim, id_num)
    
  def forward(self, feature):
    logits = self.class_head(feature)
    return logits


class AdaIN(nn.Module):
    def __init__(self, c_cond: int, c_h: int):
        super(AdaIN, self).__init__()
        self.c_h = c_h
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.linear_layer = nn.Linear(c_cond, c_h * 2)

    def forward(self, x: Tensor, x_cond: Tensor) -> Tensor:
        x_cond = self.linear_layer(x_cond)
        mean, std = x_cond[:, : self.c_h], x_cond[:, self.c_h :]
        mean, std = mean.unsqueeze(-1), std.unsqueeze(-1)
        x = x.transpose(1,2)  # (N,C,L)
        x = self.norm_layer(x)
        x = x * std + mean
        x = x.transpose(1,2)  # (N,L,C)
        return x


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
  def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None) -> None:
     super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
     self.norm1 = StyleAdaptiveLayerNorm(d_model, d_model)
     self.norm2 = StyleAdaptiveLayerNorm(d_model, d_model)
     self.norm3 = StyleAdaptiveLayerNorm(d_model, d_model)
  
  def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
              tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
              style_code=None) -> Tensor:
      r"""Pass the inputs (and mask) through the decoder layer.

      Args:
          tgt: the sequence to the decoder layer (required).
          memory: the sequence from the last layer of the encoder (required).
          tgt_mask: the mask for the tgt sequence (optional).
          memory_mask: the mask for the memory sequence (optional).
          tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
          memory_key_padding_mask: the mask for the memory keys per batch (optional).

      Shape:
          see the docs in Transformer class.
      """
      # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

      x = tgt
      if self.norm_first:
          x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
          x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
          x = x + self._ff_block(self.norm3(x))
      else:
          x = x + self._sa_block(x, tgt_mask, tgt_key_padding_mask)
          x = self.norm1(x, style_code)
          x = x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
          x = self.norm2(x, style_code)
          x = x + self._ff_block(x)
          x = self.norm3(x, style_code)
      return x


from torch.nn.modules.container import ModuleList
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
class TransformerDecoder(nn.Transformer):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                style_code=None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         style_code=style_code)

        if self.norm is not None:
            output = self.norm(output)

        return output