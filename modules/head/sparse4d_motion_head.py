# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch, math
import torch.nn as nn

from typing import List, Optional, Union
from torch.nn.modules.normalization import *

from modules.cnn.base_module import BaseModule
from tool.utils.dist_utils import reduce_mean
from modules.cnn.transformer import *

from .sparse4d_blocks.instance_bank import *
from .sparse4d_blocks.core_blocks import *
from .sparse4d_blocks.sparse3d_embedding import *
from .target import *
from .decoder import *
from .loss.base_loss import *
from .loss.sparse4d_losses import *

from tool.runner.fp16_utils import force_fp32

__all__ = ["Sparse4DMotionHead"]


class Sparse4DMotionHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        num_reg_fcs: int = 2,
        ego_fut_mode: int = 3,
        fut_ts: int = 6,
        **kwargs,
    ):
        super(Sparse4DMotionHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        def build_module(cfg):
            cfg2 = cfg.copy()
            type = cfg2.pop("type")
            return eval(type)(**cfg2)

        self.instance_bank = build_module(instance_bank)
        self.anchor_encoder = build_module(anchor_encoder)
        self.sampler = build_module(sampler)
        self.decoder = build_module(decoder)
        self.loss_cls = build_module(loss_cls)
        self.op_config_map = {
            "temp_gnn": temp_graph_model,
            "gnn": graph_model,
            "norm": norm_layer,
            "ffn": ffn,
            "deformable": deformable_model,
            "refine": refine_layer,
        }
        self.layers = nn.ModuleList(
            [build_module(self.op_config_map.get(op)) for op in self.operation_order]
        )
        self.embed_dims = self.instance_bank.embed_dims
        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()
        
        #### planning ####
        self.ego_his_encoder = nn.Linear(4, self.embed_dims, bias=False)
        self.agent_self_attn = AgentSelfAttention(self.embed_dims, depth=2)
        self.ego_agent_cross_attn = CrossAttention(self.embed_dims, num_attn_heads=8)
        self.ego_map_cross_attn = CrossAttention(self.embed_dims, num_attn_heads=8)
        ego_fut_decoder = []
        self.num_reg_fcs = num_reg_fcs
        self.ego_fut_mode = ego_fut_mode
        self.fut_ts = fut_ts
        ego_fut_dec_in_dim = self.embed_dims * 2
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(nn.Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(nn.Linear(ego_fut_dec_in_dim, self.ego_fut_mode * self.fut_ts * 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        self.loss_plan_reg = build_module(loss_reg)

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()
    
    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)  # (1, 1220, 512)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)  # (1, 1220, 256) => (1, 1220, 512)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        agent_hs: torch.Tensor,
        map_hs: torch.Tensor,
        data: dict,
    ):
        output = {}
        agent_query = agent_hs  # [bs, num_agent, embed_dims]
        map_query = map_hs  # [bs, num_map, embed_dims]
        ego_his_trajs = data['ego_his_trajs'].float().unsqueeze(1)
        ego_his_feats = self.ego_his_encoder(ego_his_trajs.flatten(2))
        ego_query = ego_his_feats

        # agent interaction
        agent_query = self.agent_self_attn(
            hidden_states=agent_query,
            attention_mask=None)

        # ego agent interaction
        ego_agent_query = self.ego_agent_cross_attn(
            hs_query=ego_query, 
            hs_key=agent_query,
            attention_mask=None)
        
        # ego map interaction
        if map_query is not None:
            ego_map_query = self.ego_map_cross_attn(
                hs_query=ego_agent_query, 
                hs_key=map_query,
                attention_mask=None)
            ego_feats = torch.cat([ego_agent_query, ego_map_query], dim=-1)  # [B, 1, 2D]
        else:
            ego_feats = torch.cat([ego_agent_query, ego_agent_query], dim=-1)

        outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], self.ego_fut_mode, self.fut_ts, 2)
        output['ego_fut_preds'] = outputs_ego_trajs
        
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data):
        output = {}
        ego_fut_cmd = data['ego_fut_cmd']
        ego_fut_masks = data['ego_fut_masks']
        ego_fut_gt = data['ego_fut_trajs'].float()
        ego_fut_preds = model_outs['ego_fut_preds']
        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )
        if math.isnan(loss_plan_l1):
            # loss_plan_l1 = torch.tensor(0.0).to(loss_plan_l1.device)
            print('ego_fut_masks: ', ego_fut_masks)
            print('ego_fut_gt: ', data['ego_fut_trajs'].float())
            print('ego_fut_preds: ', ego_fut_preds)
            print('ego_fut_cmd: ', ego_fut_cmd)
        output['loss_plan_reg'] = loss_plan_l1

        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[
            dn_valid_mask
        ]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[
            dn_valid_mask
        ][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, data):
        bs = data['ego_fut_cmd'].shape[0]
        outputs = []
        ego_fut_preds = model_outs['ego_fut_preds']
        ego_fut_cmd = data['ego_fut_cmd']
        cmd_idx = torch.nonzero(ego_fut_cmd)[:, -1]
        ego_fut_preds = ego_fut_preds[torch.arange(bs), cmd_idx, ...]
        for i in range(bs):
            outputs.append({'plan_traj': ego_fut_preds[i],})
        return outputs
