import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ReconLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss()
    def forward(self, pred, label):
        loss = self.criterion(pred, label)
        return loss


class L2SimilarityLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss()
    def forward(self, feature1, feature2):
        loss = self.criterion(feature1, feature2)
        return loss


class ContentContrastiveLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.margin = cfg.content_contrastive_loss.margin
    
    def forward(self, content_code, audio_feature):
        content_code = content_code.view(-1, content_code.shape[1]*content_code.shape[2])  # need debug???
        audio_feature = audio_feature.view(-1, audio_feature.shape[1]*audio_feature.shape[2])
        # norm
        content_code = F.normalize(content_code, p=2, dim=1)
        audio_feature = F.normalize(audio_feature, p=2, dim=1)
        # sorce
        tmp = content_code.expand(content_code.size(0), content_code.size(0), content_code.size(1)).transpose(0,1)
        scores = torch.norm(tmp-audio_feature, p=2, dim=2)

        diagonal_dist = scores.diag()

        cost_s = (self.margin - scores).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        mask = mask.to(content_code.device)
        cost_s = cost_s.masked_fill_(mask, 0)
        loss = (torch.sum(cost_s ** 2) + torch.sum(diagonal_dist ** 2)) / (2 * content_code.size(0))
        return loss


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, latent_features, logit_scale):
        device = image_features.device
        image_features = image_features.view(image_features.size(0)*image_features.size(1), -1)
        latent_features = latent_features.view(latent_features.size(0)*latent_features.size(1), -1)
        image_features = F.normalize(image_features, dim=-1)
        latent_features = F.normalize(latent_features, dim=-1)
        if self.world_size > 1:
            all_image_features, all_latent_features = gather_features(
                image_features, latent_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_latent_features.T
                logits_per_latent = logit_scale * latent_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_latent_features.T
                logits_per_latent = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ latent_features.T
            logits_per_latent = logit_scale * latent_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_latent, labels)
                     ) / 2
        return total_loss


class CTCLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
    def forward(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        N, T, C = logits.shape
        input_lengths = torch.tensor([T]*N, device=logits.device)
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)
        loss = F.ctc_loss(log_probs, flattened_targets, input_lengths, target_lengths, blank=0, reduction='sum', zero_infinity=False)
        return loss


# class GRLAdvLoss(nn.Module):
#     def __init__(self, cfg) -> None:
#         super().__init__()
#         alpha = cfg.content_grl_loss.alpha
#         from models.lib.grl_module import GradientReversal
#         self.net = nn.Sequential(
#             GradientReversal(alpha=alpha),
#             nn.Linear(cfg.feature_dim, cfg.feature_dim)
#         )
#     def forward(self, audio_feature, style_code):
#         audio_code = torch.mean(audio_feature, dim=-2)
#         audio_style_out = self.net(audio_code)
#         loss = F.l1_loss(audio_style_out, style_code)
#         return loss

class IDClassLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    def forward(self, pred, target):
        loss = self.cross_entropy_loss(pred, target)
        return loss

class L2SoftmaxLoss(nn.Module):
    def __init__(self):
        super(L2SoftmaxLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.L2loss = nn.MSELoss()
        self.label = None

    def forward(self, x):
        out = self.softmax(x)
        self.label = Variable(torch.ones(out.size()).float() * (1 / x.size(1)), requires_grad=False).to(x.device)
        loss = self.L2loss(out, self.label)
        return loss


class ComposeLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.recon_loss = ReconLoss()
        self.sim_loss = L2SimilarityLoss()
        self.content_contrastive_loss = ContentContrastiveLoss(cfg)
        self.content_CLIP_loss = ClipLoss()
        self.ctc_loss = CTCLoss(cfg)
        # self.grl_adv_loss = GRLAdvLoss(cfg)
        self.content_grl_loss = IDClassLoss()
        self.style_cls_loss = IDClassLoss()
        self.content_cls_loss = L2SoftmaxLoss()
    
    def forward(self, output, label, text_label=None, id_label=None):
        device = label.device
        content_code, style_code, audio_feature, out_content, out_audio, content_logits, audio_logits, content_grl_logits, style_logits, content_cls_logits, audio_cls_logits, logit_scale = output
        # cal loss
        loss = 0
        "rec loss"
        content_recon_loss = self.cfg.recon_loss.w*self.recon_loss(out_content, label)
        loss += content_recon_loss
        audio_recon_loss = self.cfg.recon_loss.w*self.recon_loss(out_audio, label)
        loss += audio_recon_loss
        "content code sim loss"
        if self.cfg.content_code_sim_loss.use:
            content_code_sim_loss = self.cfg.content_code_sim_loss.w*self.sim_loss(content_code, audio_feature) # if self.cfg.content_code_sim_loss.use else torch.tensor(0., dtype=torch.float32).to(device)
            loss += content_code_sim_loss
        else:
            content_code_sim_loss = torch.tensor(0., dtype=torch.float32).to(device)
        "content contrastive loss"
        if self.cfg.content_contrastive_loss.use:
            content_contrastive_loss = self.cfg.content_contrastive_loss.w*self.content_contrastive_loss(content_code, audio_feature) # if self.cfg.content_contrastive_loss.use else torch.tensor(0., dtype=torch.float32).to(device)
            loss += content_contrastive_loss
        else:
            content_contrastive_loss = torch.tensor(0., dtype=torch.float32).to(device)
        "content clip contrastive loss"
        if self.cfg.content_clip_loss.use:
            content_clip_loss = self.cfg.content_clip_loss.w*self.content_CLIP_loss(content_code, audio_feature, logit_scale) # logit_scale=tensor(14.2857, device='cuda:0', grad_fn=<ExpBackward0>)
            loss += content_clip_loss
        else:
            content_clip_loss = torch.tensor(0., dtype=torch.float32).to(device)
        "content ctc loss"
        if self.cfg.content_ctc_loss.use:
            content_ctc_loss = self.cfg.content_ctc_loss.w*self.ctc_loss(content_logits, text_label) # if self.cfg.content_ctc_loss.use else torch.tensor(0., dtype=torch.float32).to(device)
            loss += content_ctc_loss
            audio_ctc_loss = self.cfg.content_ctc_loss.w*self.ctc_loss(audio_logits, text_label) # if self.cfg.content_ctc_loss.use else torch.tensor(0., dtype=torch.float32).to(device)
            loss += audio_ctc_loss
        else:
            content_ctc_loss = audio_ctc_loss = torch.tensor(0., dtype=torch.float32).to(device)
        "content grl loss"
        if self.cfg.content_grl_loss.use and id_label is not None:
            # content_grl_loss = self.cfg.content_grl_loss.w*self.content_grl_loss(audio_grl_logits, id_label) # if self.cfg.content_grl_loss.use and id_label is not None else torch.tensor(0., dtype=torch.float32).to(device)
            content_grl_loss = self.cfg.content_grl_loss.w*self.content_grl_loss(content_grl_logits, id_label)
            loss += content_grl_loss
            content_class_acc = (torch.max(content_grl_logits, 1)[1]==id_label).float().sum()/id_label.size(0)
        else:
            content_grl_loss = torch.tensor(0., dtype=torch.float32).to(device)
            content_class_acc = torch.tensor(0., dtype=torch.float32).to(device)
        "style class loss"
        if self.cfg.style_class_loss.use and id_label is not None:
            style_class_loss = self.cfg.style_class_loss.w*self.style_cls_loss(style_logits, id_label) # if self.cfg.style_class_loss.use and id_label is not None else torch.tensor(0., dtype=torch.float32).to(device)
            loss += style_class_loss
            style_pred_acc = (torch.max(style_logits, 1)[1]==id_label).float().sum()/id_label.size(0)  # 0. or 1.
        else:
            style_class_loss = torch.tensor(0., dtype=torch.float32).to(device)
            style_pred_acc = torch.tensor(0., dtype=torch.float32).to(device)
        "content class loss"
        if self.cfg.content_class_loss.use:
            content_class_loss = self.cfg.content_class_loss.w*self.content_cls_loss(content_cls_logits)
            loss += content_class_loss
            # audio_class_loss = self.cfg.content_class_loss.w*self.content_cls_loss(audio_cls_logits)
            # loss += audio_class_loss
        else:
            content_class_loss = audio_class_loss = torch.tensor(0., dtype=torch.float32).to(device)

        # style_pred_acc = (torch.max(style_logits, 1)[1]==id_label).float().sum()/id_label.size(0) if id_label is not None else torch.tensor(0., dtype=torch.float32).to(device)  # 0. or 1.
        # content_class_acc = (torch.max(content_grl_logits, 1)[1]==id_label).float().sum()/id_label.size(0) if id_label is not None else torch.tensor(0., dtype=torch.float32).to(device)  # 0. or 1.
        # audio_class_acc = torch.tensor(float(torch.max(audio_cls_logits, 1)[1]==id_label)) if id_label is not None else torch.tensor(0., dtype=torch.float32).to(device)  # 0. or 1.

        return {
            'loss': loss,
            'content_recon_loss': content_recon_loss,
            'audio_recon_loss': audio_recon_loss,
            'content_code_sim_loss': content_code_sim_loss,
            'content_contrastive_loss': content_contrastive_loss,
            'content_clip_loss': content_clip_loss,
            'content_ctc_loss': content_ctc_loss,
            'audio_ctc_loss': audio_ctc_loss,
            'content_grl_loss': content_grl_loss,
            'style_class_loss': style_class_loss,
            'style_pred_acc': style_pred_acc,
            # 'content_class_loss': content_class_loss,
            # 'audio_class_loss': audio_class_loss,
            'content_class_acc': content_class_acc,
            # 'audio_class_acc': audio_class_acc,
        }


class ComposeCycleLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.recon_loss = ReconLoss()
        self.sim_loss = L2SimilarityLoss()
        self.content_contrastive_loss = ContentContrastiveLoss(cfg)
        self.content_CLIP_loss = ClipLoss()
        self.content_contrastive_cycle_loss = ContentContrastiveLoss(cfg)
        self.ctc_loss = CTCLoss(cfg)
        # self.grl_adv_loss = GRLAdvLoss(cfg)
        self.content_grl_loss = IDClassLoss()
        self.style_cls_loss = IDClassLoss()
        self.content_cls_loss = L2SoftmaxLoss()
    
    def forward(self, output, label, text_label=None, id_label=None):
        device = label.device
        content_code, style_code, audio_feature, out_content, out_audio, \
            content_logits, audio_logits, content_grl_logits, style_logits, \
                content_cls_logits, audio_cls_logits, \
                    style_code_shuffle, style_code_cycle, content_code_cycle, \
                        logit_scale = output
        # cal loss
        loss = 0
        "rec loss"
        content_recon_loss = self.cfg.recon_loss.w*self.recon_loss(out_content, label)
        loss += content_recon_loss
        audio_recon_loss = self.cfg.recon_loss.w*self.recon_loss(out_audio, label)
        loss += audio_recon_loss
        "content code sim loss"
        if self.cfg.content_code_sim_loss.use:
            content_code_sim_loss = self.cfg.content_code_sim_loss.w*self.sim_loss(content_code, audio_feature) # if self.cfg.content_code_sim_loss.use else torch.tensor(0., dtype=torch.float32).to(device)
            loss += content_code_sim_loss
        else:
            content_code_sim_loss = torch.tensor(0., dtype=torch.float32).to(device)
        "content contrastive loss"
        if self.cfg.content_contrastive_loss.use:
            content_contrastive_loss = self.cfg.content_contrastive_loss.w*self.content_contrastive_loss(content_code, audio_feature) # if self.cfg.content_contrastive_loss.use else torch.tensor(0., dtype=torch.float32).to(device)
            loss += content_contrastive_loss
        else:
            content_contrastive_loss = torch.tensor(0., dtype=torch.float32).to(device)
        "content clip contrastive loss"
        if self.cfg.content_clip_loss.use:
            content_clip_loss = self.cfg.content_clip_loss.w*self.content_CLIP_loss(content_code, audio_feature, logit_scale) # logit_scale=tensor(14.2857, device='cuda:0', grad_fn=<ExpBackward0>)
            loss += content_clip_loss
        else:
            content_clip_loss = torch.tensor(0., dtype=torch.float32).to(device)
        "content ctc loss"
        if self.cfg.content_ctc_loss.use:
            content_ctc_loss = self.cfg.content_ctc_loss.w*self.ctc_loss(content_logits, text_label) # if self.cfg.content_ctc_loss.use else torch.tensor(0., dtype=torch.float32).to(device)
            loss += content_ctc_loss
            audio_ctc_loss = self.cfg.content_ctc_loss.w*self.ctc_loss(audio_logits, text_label) # if self.cfg.content_ctc_loss.use else torch.tensor(0., dtype=torch.float32).to(device)
            loss += audio_ctc_loss
        else:
            content_ctc_loss = audio_ctc_loss = torch.tensor(0., dtype=torch.float32).to(device)
        "content grl loss"
        if self.cfg.content_grl_loss.use and id_label is not None:
            # content_grl_loss = self.cfg.content_grl_loss.w*self.content_grl_loss(audio_grl_logits, id_label) # if self.cfg.content_grl_loss.use and id_label is not None else torch.tensor(0., dtype=torch.float32).to(device)
            content_grl_loss = self.cfg.content_grl_loss.w*self.content_grl_loss(content_grl_logits, id_label)
            loss += content_grl_loss
            content_class_acc = (torch.max(content_grl_logits, 1)[1]==id_label).float().sum()/id_label.size(0)
        else:
            content_grl_loss = torch.tensor(0., dtype=torch.float32).to(device)
            content_class_acc = torch.tensor(0., dtype=torch.float32).to(device)
        "style class loss"
        if self.cfg.style_class_loss.use and id_label is not None:
            style_class_loss = self.cfg.style_class_loss.w*self.style_cls_loss(style_logits, id_label) # if self.cfg.style_class_loss.use and id_label is not None else torch.tensor(0., dtype=torch.float32).to(device)
            loss += style_class_loss
            style_pred_acc = (torch.max(style_logits, 1)[1]==id_label).float().sum()/id_label.size(0)  # 0. or 1.
        else:
            style_class_loss = torch.tensor(0., dtype=torch.float32).to(device)
            style_pred_acc = torch.tensor(0., dtype=torch.float32).to(device)
        "content class loss"
        if self.cfg.content_class_loss.use:
            content_class_loss = self.cfg.content_class_loss.w*self.content_cls_loss(content_cls_logits)
            loss += content_class_loss
            # audio_class_loss = self.cfg.content_class_loss.w*self.content_cls_loss(audio_cls_logits)
            # loss += audio_class_loss
        else:
            content_class_loss = audio_class_loss = torch.tensor(0., dtype=torch.float32).to(device)
        "style cycle loss"
        if self.cfg.style_cycle_loss.use:
            if self.cfg.style_cycle_loss.sim == 'cos':
                cycle_loss_target = torch.ones([style_code_cycle.size(0)]).to(style_code_cycle.device)
                style_cycle_loss = self.cfg.style_cycle_loss.w*F.cosine_embedding_loss(style_code_shuffle.detach(), style_code_cycle, cycle_loss_target)
            elif self.cfg.style_cycle_loss.sim == 'L1':
                style_cycle_loss = self.cfg.style_cycle_loss.w*F.l1_loss(style_code_shuffle.detach(), style_code_cycle)
            loss += style_cycle_loss
        else:
            style_cycle_loss = torch.tensor(0., dtype=torch.float32).to(device)
        "content cycle loss"
        if self.cfg.content_cycle_loss.use:
            content_cycle_loss = self.cfg.content_cycle_loss.w*self.content_contrastive_cycle_loss(content_code_cycle, content_code.detach())
            loss += content_cycle_loss
        else:
            content_cycle_loss = torch.tensor(0., dtype=torch.float32).to(device)

        # style_pred_acc = (torch.max(style_logits, 1)[1]==id_label).float().sum()/id_label.size(0) if id_label is not None else torch.tensor(0., dtype=torch.float32).to(device)  # 0. or 1.
        # content_class_acc = (torch.max(content_grl_logits, 1)[1]==id_label).float().sum()/id_label.size(0) if id_label is not None else torch.tensor(0., dtype=torch.float32).to(device)  # 0. or 1.
        # audio_class_acc = torch.tensor(float(torch.max(audio_cls_logits, 1)[1]==id_label)) if id_label is not None else torch.tensor(0., dtype=torch.float32).to(device)  # 0. or 1.

        return {
            'loss': loss,
            'content_recon_loss': content_recon_loss,
            'audio_recon_loss': audio_recon_loss,
            'content_code_sim_loss': content_code_sim_loss,
            'content_contrastive_loss': content_contrastive_loss,
            'content_clip_loss': content_clip_loss,
            'content_ctc_loss': content_ctc_loss,
            'audio_ctc_loss': audio_ctc_loss,
            'content_grl_loss': content_grl_loss,
            'style_class_loss': style_class_loss,
            'style_pred_acc': style_pred_acc,
            'content_class_loss': content_class_loss,
            # 'audio_class_loss': audio_class_loss,
            'content_class_acc': content_class_acc,
            # 'audio_class_acc': audio_class_acc,
            'style_cycle_loss': style_cycle_loss,
            'content_cycle_loss': content_cycle_loss,
        }



class ComposeLossOneHot(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.recon_loss = ReconLoss()
    
    def forward(self, output, label, text_label=None, id_label=None):
        content_code, style_code, audio_feature, dec_out_content, dec_out_audio = output
        loss = 0
        content_recon_loss = self.cfg.recon_loss.w*self.recon_loss(dec_out_content, label)
        loss += content_recon_loss
        audio_recon_loss = self.cfg.recon_loss.w*self.recon_loss(dec_out_audio, label)
        loss += audio_recon_loss
        return {
            'loss': loss,
            'content_recon_loss': content_recon_loss,
            'audio_recon_loss': audio_recon_loss
        }
