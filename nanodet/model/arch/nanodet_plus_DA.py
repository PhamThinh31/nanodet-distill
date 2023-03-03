# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import torch
from torch.autograd import Variable

from ..head import build_head
from .one_stage_detector_DA import OneStageDetectorDA
from ..backbone.strong_weak_alignment import FocalLoss, EFocalLoss


class NanoDetPlusDA(OneStageDetectorDA):
    def __init__(
        self,
        backbone,
        fpn,
        aux_head,
        head,
        detach_epoch=0,
    ):
        super(NanoDetPlusDA, self).__init__(
            backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head
        )
        self.aux_fpn = copy.deepcopy(self.fpn)
        self.aux_head = build_head(aux_head)
        # self.contour_head = build_head(contour_head)
        self.detach_epoch = detach_epoch
        self._current_fx_name=None
        self.log = None
        self.log_dict = None
        # self.ef = backbone.ef
        if backbone.ef:
            self.FL = EFocalLoss(class_num=2, gamma=backbone.gamma)
        else:
            self.FL = FocalLoss(class_num=2, gamma=backbone.gamma)

    def forward_train(self, gt_meta,is_train=True):
        img = gt_meta["img"]
        feat, out_s = self.backbone(img)
        if is_train:
            img_target = gt_meta['img_target']
            _, out_t = self.backbone(img_target)

            domain_s = Variable(torch.zeros(out_s.size(0)).long().cuda())
            dloss_s = 0.5*self.FL(out_s,domain_s)
            domain_t = Variable(torch.ones(out_t.size(0)).long().cuda())
            dloss_t = 0.5 * self.FL(out_t, domain_t)

        fpn_feat = self.fpn(feat)
        if self.epoch >= self.detach_epoch:
            aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
            dual_fpn_feat = (
                torch.cat([f.detach(), aux_f], dim=1)
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        else:
            aux_fpn_feat = self.aux_fpn(feat)
            dual_fpn_feat = (
                torch.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        head_out = self.head(fpn_feat)
        aux_head_out = self.aux_head(dual_fpn_feat)
        # ct_head_out = self.contour_head(fpn_feat)
        loss, loss_states,  batch_assign_res = self.head.loss(head_out, gt_meta, aux_preds=aux_head_out)#, ct_preds=ct_head_out)
        if is_train:
            loss += (dloss_s + dloss_t)
        return head_out, feat, loss, loss_states, batch_assign_res
