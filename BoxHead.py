import torch
import torch.nn.functional as F
from torch import nn
from utils import *
import torchvision
from dataset import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BoxHead(torch.nn.Module):
    def __init__(self, Classes=3, P=7, device="cpu"):
        super(BoxHead, self).__init__()
        self.device = device
        self.C = Classes
        self.P = P
        # TODO initialize BoxHead
        self.intermediate = nn.Sequential(
            nn.Linear(in_features=256 * P * P, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU()
        ).to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=Classes + 1),
            # nn.Softmax(dim=1)
        ).to(self.device)

        self.regressor = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4 * Classes)
        ).to(self.device)

        self.img_size = (800, 1088)

    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self, proposals, gt_labels, bbox, iou_thresh=0.4):
        b = len(proposals)
        labels = []
        regressor_target = []
        for i in range(b):
            iou = IOU(proposals[i], bbox[i], xaya=True)  # per_img_proposal * n_obj
            max_iou, arg_max_iou = torch.max(iou, dim=1)
            # arg_max_iou += 1
            label = torch.index_select(gt_labels[i], 0, arg_max_iou)
            bboxes = torch.index_select(bbox[i], 0, arg_max_iou)  # per_img_proposal * 4

            target_bbox = torch.zeros_like(bboxes)
            x_p = (proposals[i][:, 0] + proposals[i][:, 2]) / 2
            y_p = (proposals[i][:, 1] + proposals[i][:, 3]) / 2
            w_p = -proposals[i][:, 0] + proposals[i][:, 2]
            h_p = -proposals[i][:, 1] + proposals[i][:, 3]
            target_bbox[:, 0] = (bboxes[:, 0] - x_p) / w_p
            target_bbox[:, 1] = (bboxes[:, 1] - y_p) / h_p
            target_bbox[:, 2] = torch.log(bboxes[:, 2] / w_p)
            target_bbox[:, 3] = torch.log(bboxes[:, 3] / h_p)

            background = (max_iou > iou_thresh).int()
            label = label * background
            labels.append(label)
            regressor_target.append(target_bbox)
        labels = torch.cat(labels, dim=0).unsqueeze(1)
        regressor_target = torch.cat(regressor_target, dim=0)
        return labels, regressor_target

    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, 
                               box_regression, 
                               proposals, 
                               gt_boxes=None,
                               conf_thresh=0.5, 
                               keep_num_preNMS=500, 
                               keep_num_postNMS=100,
                               train=False):
        b = len(proposals)
        boxes = []
        scores = []
        labels = []
        start = 0
        for i in range(b):
            cls = class_logits[start: start + len(proposals[i])]
            reg_box = box_regression[start: start + len(proposals[i])]
            start += len(proposals[i])
            # get the corresponding category and bounding boxes
            # remove all the background boxes
            cls, cate = torch.max(cls, dim=1)
            nonbg_idx = torch.where(cate > 0)
            cate = cate[nonbg_idx]
            cls = cls[nonbg_idx]
            reg_box = reg_box[nonbg_idx]
            prop = proposals[i][nonbg_idx]
            reg_cls_box = torch.zeros(len(cls), 4).cuda()
            for j in range(len(cls)):
                reg_cls_box[j] = reg_box[j, 4 * cate[j] - 4: 4 * cate[j]]
            box = output_decodingd(reg_cls_box, prop)

            if train:
                # remove cross boundary boxes
                inside_idx = torch.where((box[:, 0] >= 0) & (box[:, 1] >= 0) & (box[:, 2] < self.img_size[1]) & (box[:, 3] < self.img_size[0]))
                box = box[inside_idx]
                cls = cls[inside_idx]
                cate = cate[inside_idx]

                # keep boxes that has >0.5 iou with gt boxes
                ious = IOU(box, gt_boxes[i], xaya=True)
                max_iou, _ = torch.max(ious, dim=1)

                selected_idx = torch.where(max_iou > conf_thresh)
                cls = cls[selected_idx]
                box = box[selected_idx]
                cate = cate[selected_idx]
            else:
                # testing crop the cross boundary boxes
                box[:, 0] = torch.clamp_min(box[:, 0], 0)
                box[:, 1] = torch.clamp_min(box[:, 1], 0)
                box[:, 2] = torch.clamp_max(box[:, 2], self.img_size[1])
                box[:, 3] = torch.clamp_max(box[:, 3], self.img_size[0])

            # sort boxes according to the confidence scores
            cls, sorted_idx = torch.sort(cls, descending=True)
            box = box[sorted_idx][:keep_num_preNMS]
            cate = cate[sorted_idx][:keep_num_preNMS]
            cls = cls[:keep_num_preNMS]

            # nms for different classes
            nms_indices = torchvision.ops.batched_nms(box, cls, cate, conf_thresh)
            box = box[nms_indices][:keep_num_postNMS]
            cate = cate[nms_indices][:keep_num_postNMS]
            cls = cls[nms_indices][:keep_num_postNMS]

            boxes.append(box)
            scores.append(cls)
            labels.append(cate)
        return boxes, scores, labels

    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150, eval=False):
        class_criterion = nn.CrossEntropyLoss(reduction='sum')
        regr_criterion = nn.SmoothL1Loss(reduction='sum')

        bg_labels_idx = torch.nonzero(labels==0)[:, 0]
        nonbg_labels_idx = torch.nonzero(labels!=0)[:, 0]
        # print(len(nonbg_labels_idx))
        if not eval:
            max_nonbg_sample_size = int(0.75*effective_batch)
            nonbg_sample_size = min(len(nonbg_labels_idx), max_nonbg_sample_size)
            bg_sample_size = effective_batch - nonbg_sample_size

            bg_sample_idx = bg_labels_idx[torch.randperm(len(bg_labels_idx))[:bg_sample_size]]
            nonbg_sample_idx = nonbg_labels_idx[torch.randperm(len(nonbg_labels_idx))[:nonbg_sample_size]]
            loss_class = class_criterion(
                        class_logits[torch.hstack([bg_sample_idx, nonbg_sample_idx])],
                        labels[torch.hstack([bg_sample_idx, nonbg_sample_idx])].squeeze(1))
            
            lab = labels[nonbg_sample_idx] - 1
            preds = torch.zeros_like(regression_targets[nonbg_sample_idx])
            t_preds = box_preds[nonbg_sample_idx].reshape(-1, 3, 4)
            for i in range(len(preds)):
                preds[i] = t_preds[i][lab[i]]
            loss_regr = regr_criterion(preds, regression_targets[nonbg_sample_idx])
            loss_class = loss_class / effective_batch
            loss_regr = loss_regr / nonbg_sample_size
            loss = loss_class + l*loss_regr
        else:
            loss_class = class_criterion(class_logits, labels.squeeze(1))
            lab = labels[nonbg_labels_idx] - 1
            preds = torch.zeros_like(regression_targets[nonbg_labels_idx])
            t_preds = box_preds[nonbg_labels_idx].reshape(-1, 3, 4)
            for i in range(len(preds)):
                preds[i] = t_preds[i][lab[i]]
            loss_regr = regr_criterion(preds, regression_targets[nonbg_labels_idx])
            loss_class /= len(class_logits)
            loss_regr /= len(nonbg_labels_idx)
            loss = loss_class + l*loss_regr
        return loss, loss_class, loss_regr

    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors, eval=False):
        intermediate_feature = self.intermediate(feature_vectors)
        box_pred = self.regressor(intermediate_feature)

        if not eval:
            class_logits = self.classifier(intermediate_feature)
        else:
            class_logits = self.classifier(intermediate_feature)
            class_logits = F.softmax(class_logits, dim=1)
        return class_logits, box_pred

    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    def MultiScaleRoiAlign(self, fpn_feat_list, proposals, P=7):
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
        b = len(proposals)
        feature_vectors = []
        for i in range(b):
            feature = torch.zeros(len(proposals[i]), 256, P, P)
            for j in range(len(proposals[i])):
                w = proposals[i][j, 2] - proposals[i][j, 0]
                h = proposals[i][j, 3] - proposals[i][j, 1]
                k = torch.clamp(torch.floor(4 + torch.log2(torch.sqrt(w * h) / 224)).int() - 2, 0, 4)
                feat = fpn_feat_list[k][i].unsqueeze(0)
                scales = feat.shape[3] / self.img_size[1]
                box = [proposals[i][j].unsqueeze(0)]
                feature[j] = torchvision.ops.roi_align(feat, box, output_size=P, spatial_scale=scales)
            feature_vectors.append(feature)
        feature_vectors = torch.cat(feature_vectors, dim=0)
        feature_vectors = feature_vectors.view(len(feature_vectors), -1)
        assert feature_vectors.shape[1] == 256 * P * P
        assert len(feature_vectors.shape) == 2
        return feature_vectors.cuda()

    def MultiScaleRoiAlign_mask(self, fpn_feat_list, proposals, P=14):
        b = len(proposals)
        feature_vectors = []
        for i in range(b):
            feature = torch.zeros(len(proposals[i]), 256, P, P)
            for j in range(len(proposals[i])):
                w = proposals[i][j, 2] - proposals[i][j, 0]
                h = proposals[i][j, 3] - proposals[i][j, 1]
                k = torch.clamp(torch.floor(4 + torch.log2(torch.sqrt(w * h) / 224)).int() - 2, 0)
                feat = fpn_feat_list[k][i].unsqueeze(0)
                scales = feat.shape[3] / self.img_size[1]
                box = [proposals[i][j].unsqueeze(0)]
                feature[j] = torchvision.ops.roi_align(feat, box, output_size=P, spatial_scale=scales)
            feature_vectors.append(feature)
        feature_vectors = torch.cat(feature_vectors, dim=0)
        # feature_vectors = feature_vectors.view(len(feature_vectors), -1)
        assert feature_vectors.shape[-1] == P
        assert feature_vectors.shape[1] == 256
        assert len(feature_vectors.shape) == 4
        return feature_vectors.cuda()

    def renew_proposals(self, proposals, box_pred, class_logits):
        # proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
        # return: list: len(bz) {per_image_proposals, int}
        b = len(proposals)
        start = 0
        boxes = []

        for i in range(b):
            cls = class_logits[start: start + len(proposals[i])]
            reg_box = box_pred[start: start + len(proposals[i])]
            prop = proposals[i]
            start += len(proposals[i])

            _, cls = torch.max(cls, dim=1)
            cls = cls - 1  # 0, 1, 2
            # decode the boxes
            # import pdb
            # pdb.set_trace()
            reg_cls = torch.zeros_like(prop)
            reg_box = reg_box.reshape(-1, 3, 4)
            for j in range(len(reg_cls)):
                reg_cls[j] = reg_box[j, cls[j]]
            box = output_decodingd(reg_cls, prop)  # x1 y1 x2 y2 type
            box_num = len(box)
            new_box = torch.zeros_like(box)
            # print(box[:, 0].shape)
            new_box[:, 0] = torch.max(box[:, 0], torch.zeros((1, len(box))).to(self.device))
            new_box[:, 1] = torch.max(box[:, 1], torch.zeros((1, len(box))).to(self.device))
            new_box[:, 2] = torch.min(box[:, 2], 1088 * torch.ones((1, len(box))).to(self.device))
            new_box[:, 3] = torch.min(box[:, 3], 800 * torch.ones((1, len(box))).to(self.device))
            boxes.append(new_box)
        return boxes

def plot_result():
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, bboxes_path, labels_path]

    dataset = BuildDataset(paths)
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    print("Data Loading")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    test_build_loader = BuildDataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = test_build_loader.loader()
    device="cuda:0"
    rcnn_net = BoxHead(device=device)
    rpn = RPNHead(device=device)
    rpn.load_state_dict(torch.load("./train_result/rpn/rpn_best_model.pth"))
    rpn.eval()

    rcnn_net.load_state_dict(torch.load("./train_result/boxhead_rpn_nms/boxhead_best_model.pth"))
    rcnn_net.eval()
    for idx, data_batch in enumerate(test_loader):
        images = data_batch['img'].to(device)
        bbox = data_batch["bbox"]
        labels = data_batch["labels"]
        bbox = [b.cuda() for b in bbox]
        labels = [l.cuda() for l in labels]
        _, new_coord_list, X = rpn.forward_test(images)
        proposals=[proposal[0:1000,:].detach() for proposal in new_coord_list]
        # plot(proposals, data_batch)
        fpn_feat_list= [t.detach() for t in list(X.values())]
        feature_vectors = rcnn_net.MultiScaleRoiAlign(fpn_feat_list, proposals, P=rcnn_net.P)
        class_logits, box_pred = rcnn_net(feature_vectors, eval=True)
        boxes, scores, labels = rcnn_net.postprocess_detections(class_logits, box_pred, proposals)

        img = images[0, :, :, :].cpu()
        img = transforms.functional.normalize(img, [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img.permute(1, 2, 0))

        boxes = boxes[0].cpu().detach().numpy()
        col = ["y", "g", "b"]
        for i in range(len(boxes)):
            b = boxes[i]
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3] - b[1], fill=False,
                                        color=col[labels[0][i] - 1])
            ax.add_patch(rect)
        
        for b in bbox[0]:
            b = b.cpu().detach().numpy()
            col = "r"
            rect = patches.Rectangle((b[0] - b[2]/2, b[1] - b[3]/2), b[2], b[3], fill=False,
                                        color=col)
            ax.add_patch(rect)

        plt.savefig(f"./predict_vis/boxhead/postNMS/{idx}.png")
        plt.close()
        if idx > 20:
            break

if __name__ == "__main__":
    plot_result()