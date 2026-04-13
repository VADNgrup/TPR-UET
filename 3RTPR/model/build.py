from losses import objectives
from losses import ema_loss
from model.clip_model import QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import model.layer as LocalLayer
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math


    
class DATPS(nn.Module):
    def __init__(self, args, num_classes=11003, name="a"):
        super().__init__()
        self.args = args
        self.name = name
        self.num_classes = num_classes
        self.use_token_selection = self.args.image_encoder.local_branch.enable

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.image_encoder.name, args.image_encoder.img_size, args.image_encoder.stride_size, download_root=args.iocfg.datadir)
        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        for _, v in self.base_model.visual.conv1.named_parameters():
            v.requires_grad_(False)


        self.embed_dim = self.cls_embed_dim = base_cfg['embed_dim']
        self.sratio =  self.args.image_encoder.local_branch.selection_ratio
        self.vtselection = LocalLayer.VisualFusedEmbeddingLayer(input_dim=768, embed_dim=args.image_encoder.local_branch.dim, ratio=self.sratio)
        self.ttselection = LocalLayer.TexualFusedEmbeddingLayer(input_dim=512, embed_dim=args.image_encoder.local_branch.dim, ratio=self.sratio)


        
        self.logit_scale = torch.ones([]) * (1 / args.image_encoder.temperature)

        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)
        self.vision_patch_size = base_cfg['vision_patch_size']

    #######################################    METHOD SECTION    ####################################################

    def cross_former(self, q, k, v, **kwargs):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]

        x = x.permute(1, 0, 2)  # NumxLengthxDim -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        return x


    def encode_image(self, image, att_ret=False):
        x, att, org_x = self.base_model.encode_image(image)
        if att_ret: 
            return x, att
        x = x[:, 0, :].float()
        return x
    def encode_text(self, text, att_ret=False):
        x, att, org_x = self.base_model.encode_text(text.long())
        if att_ret: 
            return x, att
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        return x
    
    def encode_image_fuse(self, image):
        x,atten_i, org_x = self.base_model.encode_image(image)
        x, _  = self.vtselection(x, org_x, atten_i) 
        return x.float()
    def encode_text_fuse(self, text):
        x,atten_t, org_x = self.base_model.encode_text(text.long())
        x, _ = self.ttselection(x, org_x, text.long(), atten_t)
        return x.float()


    ###MAIN Forward function
    def forward(self, batch):
        #data parse
        images = batch[f'images_{self.name}']
        caption_ids = batch['caption_ids']
        #text augmented input
        if self.args.erpt > 0: caption_ids = batch[f'masked_caption_ids_{self.name}']
            
        #Encode
        #//G
        image_feats, text_feats, image_attscore, text_attscore, image_rfeatures, text_rfeatures = self.base_model(images, caption_ids) #torch.Size([B, tokens, 512]) torch.Size([1, tokens, 512])
        image_fused_feats, li_feats = self.vtselection(image_feats, image_rfeatures, image_attscore)
        text_fused_feats,  lt_feats = self.ttselection(text_feats, text_rfeatures, caption_ids, text_attscore)
           
        logit_scale = self.logit_scale

        return {
            "logit_scale": logit_scale,
            "image_norms_fused_feats" : image_fused_feats / image_fused_feats.norm(dim=-1, keepdim=True), #if self.use_token_selection else None,
            "text_norms_fused_feats" : text_fused_feats / text_fused_feats.norm(dim=-1, keepdim=True),   #if self.use_token_selection else None,
        }


def build_model(args, num_classes=11003, name='a'):
    model = DATPS(args, num_classes, name)
    # covert model to fp16
    convert_weights(model)
    return model