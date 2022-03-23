from os import remove
import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import h5py
from torch.optim import Adam
import numpy as np
from torchvision.models.detection.image_list import ImageList
import time

from utils import *
from pretrained_models import pretrained_models_680

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


class MaskHead(torch.nn.Module):
    def __init__(self, Classes=3, P=14, device=None):
        super(MaskHead, self).__init__()
        self.C = Classes
        self.P = P
        self.device = device
        # TODO initialize MaskHead
        self.net = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, self.C, 1),
            nn.Sigmoid()
        )

    # This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
    # and create the ground truth for the Mask Head
    #
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
    #       proposals: list:len(bz){(per_image_proposals,4)} ——[from rpn] [x1, y1, x2, y2]
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       masks: list:len(bz){(n_obj,800,1088)}
    #       IOU_thresh: scalar (threshold to filter regressed with low IOU with a bounding box)
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
    #       gt_masks: list:len(bz){(post_NMS_boxes_per_image,2*P,2*P)}
    def preprocess_ground_truth_creation(self, mask_pred, class_logits, box_regression, proposals, gt_labels_raw, bbox,
                                         masks,
                                         IOU_thresh=0.5, keep_num_preNMS=1000, keep_num_postNMS=100):
        b = len(proposals)
        boxes = []
        scores = []
        gt_masks = []
        gt_labels = []
        start = 0
        for i in range(b):
            # cls for per image
            cls = class_logits[start: start + len(proposals[i])]
            conf, cls = torch.max(cls, dim=1)
            reg_box = box_regression[start: start + len(proposals[i])]
            box_num_image = len(reg_box)
            start += len(proposals[i])
            prop = proposals[i]

            # decode the boxes
            reg_cls = torch.zeros_like(prop)
            reg_box = reg_box.reshape(-1, 3, 4)
            for j in range(len(reg_cls)):
                reg_cls[j] = reg_box[j, cls[j] - 1]
            # box for per image
            box = output_decodingd(reg_cls, prop)  # x1 y1 x2 y2 type

            gt_masks_image = torch.zeros((box_num_image, 2 * self.P, 2 * self.P))
            gt_labels_image = torch.zeros((box_num_image, 1))
            for c_ in range(self.C):
                c = c_ + 1
                indices = torch.nonzero(cls == c).squeeze(1)  # cls(0, 1, 2, 3)
                gt_indices = torch.nonzero(gt_labels_raw[i] == c).squeeze(1)  # gt_labels_raw(0, 1, 2, 3)
                if len(indices) == 0 or len(gt_indices) == 0:
                    continue
                for indice, box_class in zip(indices, box[indices]):
                    # import pdb
                    # pdb.set_trace()
                    # assign gt mask to a single box
                    left_box = box_class
                    # choose the gt_box which has the max iou with the proposed box
                    # print(left_box.is_cuda)
                    # print(bbox[i][gt_indices].is_cuda)
                    iou_res, max_iou_bbox_idx = max_IOU(left_box.unsqueeze(0), bbox[i][gt_indices], xaya=True)
                    # get the mask
                    gt_mask_perbox = masks[i][gt_indices[max_iou_bbox_idx]]  # (1, 800, 1088)
                    # intersection of the mask and proposed box
                    gt_mask_perbox = gt_mask_perbox[None, :, int(left_box[1]):int(left_box[3]),
                                     int(left_box[0]):int(left_box[2])]
                    # resize(Note interpolate only receive (bz, c, x, y) not (x, y))
                    if gt_mask_perbox.shape[-1] == 0 or gt_mask_perbox.shape[-2] == 0:
                        gt_mask_perbox = torch.zeros((2 * self.P, 2 * self.P))
                    else:
                        gt_mask_perbox = F.interpolate(gt_mask_perbox, size=(2 * self.P, 2 * self.P))[0, 0]
                    gt_masks_image[indice] = gt_mask_perbox
                    gt_labels_image[indice] = c  # gt_labels (1, 2, 3)
                    # print(gt_masks_image.shape)
            gt_masks.append(gt_masks_image)
            gt_labels.append(gt_labels_image)
            boxes.append(box)
        return boxes, scores, gt_labels, gt_masks, mask_pred

    def visualize_original(self, nxt):
        imgs, labels, masks, bboxs = nxt['img'], nxt['labels'], nxt['masks'], nxt['bbox']
        # draw
        cmap = ["viridis", "summer", "autumn"]
        color = ['b', 'g', 'r']
        label = ['vehicle', 'person', 'animal']
        N = len(imgs)
        # imgs = imgs.clone().detach().numpy()
        for i in range(0, N):
            fig, ax = plt.subplots(1, 1)
            img_ = imgs[i]
            img_ = transforms.functional.normalize(img_,
                                                   [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                   [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
            ax.imshow(img_.permute(1, 2, 0).squeeze())
            label_ = labels[i].clone().detach().numpy()
            label_cls = Counter(label_)
            # print(label_cls)
            begin_ = 0
            masks_ = masks[i]
            bbox = bboxs[i].clone().detach().numpy()

            for cls in label_cls.keys():
                cls_num = label_cls[cls]
                for mask in masks_[begin_:begin_ + cls_num]:
                    mask_ = mask.clone().detach().numpy()
                    msk = np.ma.masked_where(mask_ == 0, mask_)
                    ax.imshow(msk.squeeze(), cmap=cmap[int(cls) - 1], alpha=0.7)
                for box in bbox[begin_:begin_ + cls_num]:
                    rect = patches.Rectangle((box[0] - box[2] / 2, box[1] - box[3] / 2), box[2], box[3],
                                             edgecolor=color[int(cls) - 1],
                                             facecolor="none")
                    ax.add_patch(rect)
                    ax.text(box[0] - box[2] / 2, (box[1] - box[3] / 2) - 5, label[int(cls) - 1], fontsize=8,
                            c=color[int(cls) - 1])
                begin_ += cls_num
            # plt.show()

    def visualize_mask(self, imgs, boxes, labels, masks, gt=False, image_size=(800, 1088)):
        # imgs: bz,
        # boxes: bz, {img_prop, 4} (xyxy)
        # labels: bz, {img_prop, 1} (1, 2, 3)
        # masks: bz, {img_prop, image_size[0],image_size[1]}
        # gt_masks: bz, {img_prop,2*P,2*P}
        for i in range(len(imgs)):
            img_ = imgs[i]
            img_boxes = boxes[i]
            img_labels = labels[i]
            img_masks = masks[i]
            # draw
            cmap = ["viridis", "summer", "autumn"]
            color = ['b', 'g', 'r']
            label = ['vehicle', 'person', 'animal']
            fig, ax = plt.subplots(1, 1)
            img_ = transforms.functional.normalize(img_,
                                                   [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                   [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
            img_ = img_.detach().cpu()
            ax.imshow(img_.permute(1, 2, 0).squeeze())

            label_cls = Counter(img_labels)

            for cls in label_cls.keys():
                cls_indices = torch.nonzero(img_labels == cls)[:, 0]
                for box, mask in zip(img_boxes[cls_indices][:2], img_masks[cls_indices][:2]):
                    # import pdb
                    # pdb.set_trace()
                    if not gt:
                        mask_ = mask.clone().detach().numpy()
                        msk = np.ma.masked_where(mask_ == 0, mask_)
                    else:
                        mask_size = (int(box[3]) - int(box[1]), int(box[2]) - int(box[0]))
                        mask_ = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=mask_size)[0][0]
                        msk = torch.zeros((image_size[0], image_size[1]))
                        # mask_x2 = min(int(box[0])+mask_size[0], image_size[0])
                        # mask_y2 = min(int(box[1])+mask_size[1], image_size[1])
                        # mask_x1 = max(0, int(box[0]))
                        # mask_y1 = max(0, int(box[1]))
                        # import pdb
                        # pdb.set_trace()
                        if int(box[0]) + mask_size[0] > image_size[0] or int(box[1]) + mask_size[1] > image_size[1]:
                            continue
                        msk[int(box[1]):int(box[1]) + mask_size[0], int(box[0]):int(box[0]) + mask_size[1]] = mask_
                        msk = np.ma.masked_where(msk == 0, msk)
                    ax.imshow(msk.squeeze(), cmap=cmap[int(cls) - 1], alpha=0.7)
                    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                             edgecolor=color[int(cls) - 1],
                                             facecolor="none")
                    ax.add_patch(rect)
                    ax.text(box[0], box[1] - 5, label[int(cls) - 1], fontsize=8, c=color[int(cls) - 1])

    # general function that takes the input list of tensors and concatenates them along the first tensor dimension
    # Input:
    #      input_list: list:len(bz){(dim1,?)}
    # Output:
    #      output_tensor: (sum_of_dim1,?)
    def flatten_inputs(self, input_list):
        output_tensor = input_list[0]
        for input_tensor in input_list[1:]:
            input_tensor = input_tensor
            output_tensor = torch.cat((output_tensor, input_tensor), dim=0)
        return output_tensor

    # This function does the post processing for the result of the Mask Head for a batch of images. It project the predicted mask
    # back to the original image size
    # Use the regressed boxes to distinguish between the images
    # Input:
    #       masks_outputs: (total_boxes,C,2*P,2*P)
    #       proposals: list:len(bz){(per_image_proposals,4)} ——[from rpn] [x1, y1, x2, y2]
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
    #       image_size: tuple:len(2)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    #       projected_masks: list:len(bz){(post_NMS_boxes_per_image,image_size[0],image_size[1]
    def postprocess_mask(self, masks_outputs, proposals, box_regression, class_logits, image_size=(800, 1088),
                         keep_num_postNMS=100):
        boxes = []
        scores = []
        labels = []
        start = 0
        projected_masks = []
        conf_thresh = 0.5
        IOU_thresh = 0.5
        for i, proposal in enumerate(proposals):
            # TODO: need to get the int(boxes[0])
            cls = class_logits[start: start + len(proposals[i])]
            reg_box = box_regression[start: start + len(proposals[i])]
            mask = masks_outputs[start: start + len(proposals[i])]
            start += len(proposals[i])

            # Confidence cutoff
            non_bg = torch.nonzero(cls[:, 0] < conf_thresh).squeeze(1)
            cls = cls[non_bg]
            reg_box = reg_box[non_bg]
            prop = proposal[non_bg]
            mask = mask[non_bg]

            # get rid of all background predictions
            conf, cls = torch.max(cls, dim=1)
            nonbg_indices = torch.nonzero(cls).squeeze(1)
            conf = torch.index_select(conf, 0, nonbg_indices)
            cls = torch.index_select(cls, 0, nonbg_indices) - 1  # 0, 1, 2
            reg_box = torch.index_select(reg_box, 0, nonbg_indices)
            prop = torch.index_select(prop, 0, nonbg_indices)
            mask = torch.index_select(mask, 0, nonbg_indices)

            # decode the boxes
            reg_cls = torch.zeros_like(prop)
            reg_box = reg_box.reshape(-1, 3, 4)
            for j in range(len(reg_cls)):
                reg_cls[j] = reg_box[j, cls[j]]
            box = output_decodingd(reg_cls, prop)  # x1 y1 x2 y2 type
            # print(box)

            left_indices = []
            for c in range(self.C):
                indices = torch.nonzero(cls == c).squeeze(1)
                if len(indices) == 0:
                    continue
                box_list = list(range(len(indices)))
                left_index = []
                while len(box_list) > 0:
                    l = box_list[0]
                    remove_list = [l]
                    for b in range(1, len(box_list)):
                        iou = single_IOU(box[indices[l]], box[indices[box_list[b]]], True, True)
                        if iou > IOU_thresh:
                            remove_list.append(box_list[b])
                    for r in remove_list:
                        box_list.remove(r)
                    left_index.append(l)
                left_indices.append(indices[left_index])
            if len(left_indices) == 0:
                return [], [], []
            left_indices, _ = torch.sort(torch.cat(left_indices))
            conf = (conf[left_indices])[:keep_num_postNMS]
            cls = (cls[left_indices])[:keep_num_postNMS] + 1  # 1,2,3
            box = (box[left_indices])[:keep_num_postNMS]
            mask = (mask[left_indices])[:keep_num_postNMS]
            img_projected_masks = []
            for i, b, m in zip(range(len(box)), box, mask):
                c = cls[i]
                m = m[c - 1]
                mask_size = (int(b[2]) - int(b[0]), int(b[3]) - int(b[1]))
                if int(b[0]) + mask_size[0] > image_size[0] or int(b[1]) + mask_size[1] > image_size[1]:
                    continue
                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=mask_size)
                project_mask = torch.zeros((image_size[0], image_size[1]))
                project_mask[int(b[0]):int(b[0]) + mask_size[0], int(b[1]):int(b[1]) + mask_size[1]] = m[0][
                    0]  # (self.P, self.P)
                img_projected_masks.append(project_mask)
            img_projected_masks = torch.stack(img_projected_masks, dim=0)
            assert len(img_projected_masks.shape) == 3
            mask = torch.ones_like(img_projected_masks) * (img_projected_masks > 0.7)

            boxes.append(box)
            scores.append(conf)
            labels.append(cls)
            projected_masks.append(mask)

        return boxes, scores, labels, projected_masks

    # Compute the total loss of the Mask Head
    # Input:
    #      mask_output: (total_boxes,C,2*P,2*P)
    #      labels: (total_boxes)
    #      gt_masks: (total_boxes,2*P,2*P)
    # Output:
    #      mask_loss
    def compute_loss(self, mask_output, labels, gt_masks):
        # the total_boxes means the boxes after NMS?
        mask_loss = 0
        loss = nn.BCELoss(reduction='sum')
        for c in range(self.C):
            # import pdb
            # pdb.set_trace()
            cls = c
            cls_idx = torch.nonzero(labels == cls + 1)[:, 0]
            if len(cls_idx) == 0:
                continue
            cls_masks = mask_output[cls_idx][:, c, ...]
            gt_cls_masks = gt_masks[cls_idx]
            mask_loss += loss(cls_masks.reshape(-1, 1), gt_cls_masks.reshape(-1, 1))
        mask_loss = mask_loss / (len(mask_output) * self.P * self.P)
        return mask_loss

    # Forward the pooled feature map Mask Head
    # Input:
    #        features: (total_boxes, 256,P,P)
    # Outputs:
    #        mask_outputs: (total_boxes,C,2*P,2*P)
    def forward(self, features):
        mask_outputs = self.net(features)
        return mask_outputs

