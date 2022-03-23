from matplotlib.pyplot import grid
import torch
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, Tensor
from utils import *
import torchvision
from backbone import Resnet50Backbone


class RPNHead(torch.nn.Module):
    # The input of the initialization of the RPN is:
    # Input:
    #       computed_anchors: the anchors computed in the dataset
    #       num_anchors: the number of anchors that are assigned to each grid cell
    #       in_channels: number of channels of the feature maps that are outputed from the backbone
    #       device: the device that we will run the model
    def __init__(self, num_anchors=3, in_channels=256, device='cuda',
                 anchors_param=dict(ratio=[[1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]],
                                    scale=[32, 64, 128, 256, 512],
                                    grid_size=[(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)],
                                    stride=[4, 8, 16, 32, 64]),
                 freeze_backbone=False,
                 backbone_ckpt=None
                 ):
        ######################################
        # TODO initialize RPN
        #######################################
        super(RPNHead,self).__init__()
        self.anchors_param = anchors_param
        self.num_anchors = num_anchors
        self.in_channels = in_channels
        self.device = device
        self.len_fpn = len(anchors_param['ratio'])
        self.total_anchors = 0
        for i in range(self.len_fpn):
            self.total_anchors += self.anchors_param['grid_size'][i][0] * self.anchors_param['grid_size'][i][1] * 3
        
        self.backbone = Resnet50Backbone(checkpoint_file=backbone_ckpt, device=device, eval=freeze_backbone)
        self.intermediate = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding="same"),
                                          nn.BatchNorm2d(in_channels),
                                          nn.ReLU()).to(device)
        
        self.classifier = nn.Sequential(nn.Conv2d(in_channels, self.num_anchors, 1, 1, padding="same"),
                                        nn.Sigmoid()).to(device)
        
        self.regressor = nn.Conv2d(in_channels, 4*self.num_anchors, 1, 1, padding="same").to(device)
        self.anchors = self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict = {}
        self.backbone_keys = ['0', '1', '2', '3', 'pool']
        self.image_size = [800, 1066]
        

    # Forward each level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       X: list:len(FPN){(bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logits: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       bbox_regs: list:len(FPN){(bz,4*num_anchors, grid_size[0],grid_size[1])}
    def forward(self, X):
        X = self.backbone(X)
        logits = []
        bbox_regs = []
        for i in range(self.len_fpn):
            logit, reg = self.forward_single(X[self.backbone_keys[i]])
            logits.append(logit)
            bbox_regs.append(reg)
        return logits, bbox_regs
    
    def forward_test(self, X):
        X = self.backbone(X)
        logits = []
        bbox_regs = []
        for i in range(self.len_fpn):
            logit, reg = self.forward_single(X[self.backbone_keys[i]])
            logits.append(logit)
            bbox_regs.append(reg)
        sorted_clas_list, sorted_coord_list = self.postprocess(logits, bbox_regs)# xyxy coding
        return sorted_clas_list, sorted_coord_list, X

    # Forward a single level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       feature: (bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logit: (bz,1*num_acnhors,grid_size[0],grid_size[1])
    #       bbox_regs: (bz,4*num_anchors, grid_size[0],grid_size[1])
    def forward_single(self, feature):
        out = self.intermediate(feature)
        logit = self.classifier(out)
        bbox_reg = self.regressor(out)
        return logit, bbox_reg


    # This function creates the anchor boxes for all FPN level
    # Input:
    #       aspect_ratio: list:len(FPN){list:len(number_of_aspect_ratios)}
    #       scale:        list:len(FPN)
    #       grid_size:    list:len(FPN){tuple:len(2)}
    #       stride:        list:len(FPN)
    # Output:
    #       anchors_list: list:len(FPN){(num_anchors, grid_size[0], grid_size[1], 4)}
    # Note:
    #       anchors is in x, y, w, h form
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        len_FPN = len(aspect_ratio)
        anchor_list = []
        for i in range(len_FPN):
            anchors = self.create_anchors_single(aspect_ratio[i], scale[i], grid_sizes[i], stride[i])
            anchor_list.append(anchors)
        return anchor_list



    # This function creates the anchor boxes for one FPN level
    # Input:
    #      aspect_ratio: list:len(number_of_aspect_ratios)
    #      scale: scalar
    #      grid_size: tuple:len(2)
    #      stride: scalar
    # Output:
    #       anchors: (num_anchors, grid_size[0], grid_size[1], 4)
    def create_anchors_single(self, aspect_ratio, scale, grid_sizes, stride):
        anchors= torch.zeros((3, grid_sizes[0], grid_sizes[1], 4))
        for j in range(3):
            w = np.sqrt(aspect_ratio[j]*(scale**2))
            h = 1/np.sqrt(aspect_ratio[j]/(scale**2))
            grid_rows, grid_cols = grid_sizes[0], grid_sizes[1]
            x_span, y_span = stride, stride
            x = x_span*torch.arange(grid_cols) + x_span/2
            y = y_span*torch.arange(grid_rows) + y_span/2
            x, y = torch.meshgrid(x, y)
            anchors[j, :, :, 0] = x.permute(1, 0)
            anchors[j, :, :, 1] = y.permute(1, 0)
            anchors[j, :, :, 2] = w*torch.ones((grid_rows, grid_cols))
            anchors[j, :, :, 3] = h*torch.ones((grid_rows, grid_cols))
        assert anchors.shape == (3, grid_sizes[0], grid_sizes[1], 4)
        # anchors = anchors.reshape(-1, 4)
        return anchors

    def get_anchors(self):
        return self.anchors

    # This function creates the ground truth for a batch of images
    # Input:
    #      bboxes_list: list:len(bz){(number_of_boxes,4)}
    #      indexes: list:len(bz)
    #      image_shape: list:len(bz){tuple:len(2)}
    # Ouput:
    #      ground: list:len(FPN){(bz,num_anchors,grid_size[0],grid_size[1])}
    #      ground_coord: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    def create_batch_truth(self, bboxes_list, indexes, image_shape):
        bz = len(bboxes_list)
        ground = [[], [], [], [], []]
        ground_coord = [[], [], [], [], []]
        for i in range(bz):
            gt_clas, gt_coord = self.create_ground_truth(bboxes_list[i], 
                                                         indexes[i], 
                                                         self.anchors_param['grid_size'],
                                                         self.get_anchors(), 
                                                         image_shape)
            for t in range(self.len_fpn):
                ground[t].append(gt_clas[t])
                ground_coord[t].append(gt_coord[t])
        ground = [torch.stack(i).to(self.device) for i in ground]
        ground_coord = [torch.stack(i).to(self.device) for i in ground_coord]
        return ground, ground_coord

    # This function create the ground truth for one image for all the FPN levels
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset)
    #       grid_size:   list:len(FPN){tuple:len(2)}
    #       anchor_list: list:len(FPN){(num_anchors, grid_size[0], grid_size[1], 4)}
    # Output:
    #       ground_clas: list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
    #       ground_coord: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}
    def create_ground_truth(self, bboxes, index, grid_sizes, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        # TODO create ground truth for a single image
        #####################################################
        H, W = image_size
        gt_coord_list = []
        gt_clas_list = []

        flatten_anchors = flat_anchors(anchors)
        ground_clas = torch.zeros(flatten_anchors.shape[0], 1)
        ground_coord = torch.zeros_like(flatten_anchors)
        x1 = flatten_anchors[:, 0] - flatten_anchors[:, 2] / 2.0
        y1 = flatten_anchors[:, 1] - flatten_anchors[:, 3] / 2.0
        x2 = flatten_anchors[:, 0] + flatten_anchors[:, 2] / 2.0
        y2 = flatten_anchors[:, 1] + flatten_anchors[:, 3] / 2.0
        inside_idx = torch.where((x1 >= 0) & (y1 >= 0) & (x2 < image_size[1]) & (y2 < image_size[0]))
        iou_mat = IOU(flatten_anchors[inside_idx], bboxes)
        max_iou_box, _ = torch.max(iou_mat, dim=0)
        inside_idx = inside_idx[0]
        max_ious, max_iou_idxes = torch.max(iou_mat, dim=1)
        pos_idx = inside_idx[torch.where(max_ious > 0.7)]
        pos_box_idx = max_iou_idxes[torch.where(max_ious > 0.7)]
        neg_idx = inside_idx[torch.where(max_ious < 0.3)]

        for idx, box_idx in zip(pos_idx, pos_box_idx):
            i = idx.item()
            k = box_idx.item()
            ground_clas[i, 0] = 1

            x,y,w,h = bboxes[k]
            
            xa, ya, wa, ha = flatten_anchors[i]
            
            ground_coord[i, 0] = (x - xa) / wa
            ground_coord[i, 1] = (y - ya) / ha
            ground_coord[i, 2] = torch.log(w / wa)
            ground_coord[i, 3] = torch.log(h / ha)
        
        # negative label with IOU < 0.3
        ground_clas[neg_idx] = -1

        # positive label with highest IOU
        for k in range(bboxes.shape[0]):
            max_iou_box_idx = inside_idx[torch.where(torch.isclose(iou_mat[:, k], max_iou_box[k]))[0]]

            ground_clas[max_iou_box_idx] = 1
            one_idx = torch.ones_like(max_iou_box_idx)

            x,y,w,h = bboxes[k]
            
            xa, ya, wa, ha = flatten_anchors[max_iou_box_idx][:, 0], flatten_anchors[max_iou_box_idx][:, 1], flatten_anchors[max_iou_box_idx][:, 2], flatten_anchors[max_iou_box_idx][:, 3]
            
            ground_coord[(max_iou_box_idx, one_idx * 0)] = (x - xa) / wa
            ground_coord[(max_iou_box_idx, one_idx * 1)] = (y - ya) / ha
            ground_coord[(max_iou_box_idx, one_idx * 2)] = torch.log(w / wa)
            ground_coord[(max_iou_box_idx, one_idx * 3)] = torch.log(h / ha)
        
        start = 0
        for i in range(len(anchors)):
            level_cnt = self.num_anchors * grid_sizes[i][0] * grid_sizes[i][1]
            level_gt_coord = ground_coord[start: start + level_cnt]
            level_gt_clas = ground_clas[start: start + level_cnt]
            level_gt_coord = level_gt_coord.reshape(self.num_anchors, grid_sizes[i][0] , grid_sizes[i][1], 4)
            level_gt_coord = level_gt_coord.permute(0, 3, 1, 2).reshape(-1, grid_sizes[i][0] , grid_sizes[i][1])
            level_gt_clas = level_gt_clas.reshape(self.num_anchors, grid_sizes[i][0], grid_sizes[i][1])
            gt_coord_list.append(level_gt_coord)
            gt_clas_list.append(level_gt_clas)
            start += level_cnt

        self.ground_dict[key] = (gt_clas_list, gt_coord_list)

        return gt_clas_list, gt_coord_list

    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self, p_out, n_out):

        # torch.nn.BCELoss()
        # TODO compute classifier's loss
        sum_count = p_out.shape[0] + n_out.shape[0]
        loss = torch.nn.BCELoss(reduction='sum')
        loss_class = loss(p_out, torch.ones_like(p_out)) + loss(n_out, torch.zeros_like(n_out))
        return loss_class, sum_count

    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self, pos_target_coord, pos_out_r):
        # torch.nn.SmoothL1Loss()
        # TODO compute regressor's loss
        sum_count = pos_out_r.shape[0]
        loss = torch.nn.SmoothL1Loss(reduction='sum')
        loss_reg = loss(pos_out_r.reshape(-1, 1), pos_target_coord.reshape(-1, 1))

        return loss_reg, sum_count

    # Compute the total loss for the FPN heads
    # Input:
    #       clas_out_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       regr_out_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       targ_clas_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       targ_regr_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       l: weighting lambda between the two losses
    # Output:
    #       loss: scalar
    #       loss_c: scalar
    #       loss_r: scalar
    def compute_loss(self, clas_out_list, regr_out_list, targ_clas_list, targ_regr_list, l=1, effective_batch=75, eval=False):
        total_loss, total_loss_c, total_loss_r = 0., 0., 0.
        flat_clas_out = torch.clamp(torch.cat([i.permute(0, 2, 3, 1).reshape(-1, 1) for i in clas_out_list], dim=0), 0, 1)
        flat_regr_out = torch.cat([i.permute(0, 2, 3, 1).reshape(-1, 4) for i in regr_out_list], dim=0)
        flat_clas_targ = torch.cat([i.permute(0, 2, 3, 1).reshape(-1, 1) for i in targ_clas_list], dim=0)
        flat_regr_targ = torch.cat([i.permute(0, 2, 3, 1).reshape(-1, 4) for i in targ_regr_list], dim=0)
        pos_targ = torch.nonzero(flat_clas_targ == 1)[:, 0]
        neg_targ = torch.nonzero(flat_clas_targ == -1)[:, 0]

        if not eval:
            sample_len = int(min(len(pos_targ), effective_batch/2))
            pos_targ_sampleidx = np.random.choice(a=len(pos_targ), size=sample_len, replace=False)
            neg_targ_sampleidx = np.random.choice(a=len(neg_targ), size=effective_batch - sample_len, replace=False)
            mbatch_clas_pos = flat_clas_out[pos_targ[pos_targ_sampleidx]]
            mbatch_clas_neg = flat_clas_out[neg_targ[neg_targ_sampleidx]]
            mbatch_regr_out = flat_regr_out[pos_targ[pos_targ_sampleidx]]
            mbatch_regr_targ = flat_regr_targ[pos_targ[pos_targ_sampleidx]]
            loss_c, sum_count_c = self.loss_class(mbatch_clas_pos, mbatch_clas_neg)
            loss_r, sum_count_r = self.loss_reg(mbatch_regr_targ, mbatch_regr_out)
            loss = loss_c/sum_count_c + loss_r * l / sum_count_r
        else:
            loss_c, sum_count_c = self.loss_class(flat_clas_out[pos_targ], flat_clas_out[neg_targ])
            loss_r, sum_count_r = self.loss_reg(flat_regr_targ[pos_targ], flat_regr_out[pos_targ])
            loss = loss_c/sum_count_c + loss_r * l / sum_count_r
        total_loss += loss
        total_loss_c += loss_c/sum_count_c
        total_loss_r += loss_r * l / sum_count_r
        return total_loss, total_loss_c, total_loss_r


    # Post process for the outputs for a batch of images
    # Input:
    #       out_c: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       out_r: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinate of the boxes that the NMS kept)
    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=10000, keep_num_postNMS=1000, train=True):
        bz = len(out_c[0])
        _clas_list = []
        _prebox_list = []
        for i in range(bz):
            out_c_img = [c[i:i+1] for c in out_c]
            out_r_img = [r[i:i+1] for r in out_r]
            _clas, _prebox = self.postprocessImg(out_c_img, 
                                                out_r_img, 
                                                IOU_thresh, 
                                                keep_num_preNMS, 
                                                keep_num_postNMS,
                                                train)
            _clas_list.append(_clas)
            _prebox_list.append(_prebox)
        return _clas_list, _prebox_list

    # Post process the output for one image
    # Input:
    #      mat_clas: list:len(FPN){(1,1*num_anchors,grid_size[0],grid_size[1])}  (score of the output boxes)
    #      mat_coord: list:len(FPN){(1,4*num_anchors,grid_size[0],grid_size[1])} (encoded coordinates of the output boxess)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS, train):
        flatten_regr, flatten_clas, flatten_anchors = output_flattening(mat_coord, mat_clas, self.get_anchors())
        boxes = output_decoding(flatten_regr, flatten_anchors, device=self.device) # xyxy coding
        if train:
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            inside_idx = torch.where((x1 >= 0) & (y1 >= 0) & (x2 < self.image_size[1]) & (y2 < self.image_size[0]))
            flatten_clas = flatten_clas[inside_idx]
            boxes = boxes[inside_idx]
        else:
            boxes[:, 0] = torch.clamp_min(boxes[:, 0], 0)
            boxes[:, 1] = torch.clamp_min(boxes[:, 1], 0)
            boxes[:, 2] = torch.clamp_max(boxes[:, 2], self.image_size[1])
            boxes[:, 3] = torch.clamp_max(boxes[:, 3], self.image_size[0])
        sorted_clas, sorted_index = torch.sort(flatten_clas, descending=True)
        sorted_coord = boxes[sorted_index]
        sorted_clas = sorted_clas[:keep_num_preNMS]
        sorted_coord = sorted_coord[:keep_num_preNMS]
        # ind = torch.nonzero(sorted_clas > 0.6)[:, 0]
        # sorted_clas = sorted_clas[ind]
        # sorted_coord = sorted_coord[ind]
        nms_clas, nms_prebox = self.NMS(sorted_clas, sorted_coord, IOU_thresh)
        nms_clas = nms_clas[:keep_num_postNMS]
        nms_prebox = nms_prebox[:keep_num_postNMS]
        return nms_clas, nms_prebox

    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self, clas, coords, thresh):
        nms_indices = torchvision.ops.batched_nms(coords, clas, torch.zeros_like(clas), thresh)
        return clas[nms_indices], coords[nms_indices]

def plot_nms():
    device = "cuda:0"
    rpn = RPNHead(device=device)
    # X = torch.randn(2, 3, 800, 1088)
    # logits, bbox_regs = rpn(X)
    # print("logits shape", logits.shape)
    # print("bbox_reg shape", bbox_regs.shape)
    # print(rpn.get_anchors().shape)
    # fake_bbox = torch.tensor([[250, 320, 100, 120], [300, 400, 100, 120]])
    # rpn.create_ground_truth(fake_bbox, 0, [50, 68], rpn.get_anchors(), [800, 1088])
    rpn.load_state_dict(torch.load("./train_result/rpn_best_model.pth"))
    
    from dataset import BuildDataLoader, BuildDataset
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, bboxes_path, labels_path]
    dataset = BuildDataset(paths)
    backbone = Resnet50Backbone(device=device)
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = train_build_loader.loader()
    rpn.eval()
    for i, batch in enumerate(train_loader, 0):
        img = batch["img"].to(device)
        logits, bbox_regs = rpn.forward_test(backbone(img))
        # logits, bbox_regs = rpn(backbone(img))
        logits = logits[0]
        bbox_regs[0] = bbox_regs[0][:200]
        logits[0] = logits[0][:200]
        images = batch['img'][0, :, :, :]
        indexes = batch['idx']
        boxes = batch['bbox']
        # gt, ground_coord = rpn.create_batch_truth(boxes, indexes, images.shape[-2:])

        nms_clas_list, nms_prebox_list = rpn.postprocess(logits, bbox_regs)
        images = transforms.functional.normalize(images,
                                                 [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(images.permute(1, 2, 0))

        for j in range(len(nms_clas_list[0])):
            if nms_clas_list[0][j] > 0.5:
                col = 'b'
                coord = nms_prebox_list[0][j].cpu().detach().numpy()
                rect = patches.Rectangle((coord[0] - coord[2] / 2, coord[1] - coord[3] / 2), coord[2], coord[3], fill=False,
                                        color=col)
                ax.add_patch(rect)
        for j in range(len(boxes[0])):
            col = 'r'
            rect = patches.Rectangle((boxes[0][j, 0] - boxes[0][j, 2] / 2, boxes[0][j, 1] - boxes[0][j, 3] / 2), boxes[0][j, 2], boxes[0][j, 3],
                                     fill=False, color=col)
            ax.add_patch(rect)
        ax.title.set_text("After NMS")

        plt.savefig(f"./predict_vis/{i}.png")
        plt.close()

        if (i > 20):
            break

def plot_prenms():
    device = "cuda:0"
    rpn = RPNHead(device=device)
    # X = torch.randn(2, 3, 800, 1088)
    # logits, bbox_regs = rpn(X)
    # print("logits shape", logits.shape)
    # print("bbox_reg shape", bbox_regs.shape)
    # print(rpn.get_anchors().shape)
    # fake_bbox = torch.tensor([[250, 320, 100, 120], [300, 400, 100, 120]])
    # rpn.create_ground_truth(fake_bbox, 0, [50, 68], rpn.get_anchors(), [800, 1088])
    rpn.load_state_dict(torch.load("./train_result/rpn_best_model.pth"))
    
    from dataset import BuildDataLoader, BuildDataset
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, bboxes_path, labels_path]
    dataset = BuildDataset(paths)
    backbone = Resnet50Backbone(device=device)
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = train_build_loader.loader()
    rpn.eval()
    for i, batch in enumerate(train_loader, 0):
        img = batch["img"].to(device)
        logits, bbox_regs = rpn.forward_test(backbone(img))
        # logits, bbox_regs = rpn(backbone(img))
        bbox_regs[0] = bbox_regs[0][:200]
        logits[0] = logits[0][:200]
        images = batch['img'][0, :, :, :]
        indexes = batch['idx']
        boxes = batch['bbox']

        images = transforms.functional.normalize(images,
                                                 [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(images.permute(1, 2, 0))

        for j in range(len(logits[0])):
            if logits[0][j] > 0.5:
                col = 'b'
                coord = bbox_regs[0][j].cpu().detach().numpy()
                rect = patches.Rectangle((coord[0] - coord[2] / 2, coord[1] - coord[3] / 2), coord[2], coord[3], fill=False,
                                        color=col)
                ax.add_patch(rect)
        for j in range(len(boxes[0])):
            col = 'r'
            rect = patches.Rectangle((boxes[0][j, 0] - boxes[0][j, 2] / 2, boxes[0][j, 1] - boxes[0][j, 3] / 2), boxes[0][j, 2], boxes[0][j, 3],
                                     fill=False, color=col)
            ax.add_patch(rect)
        ax.title.set_text("After NMS")

        plt.savefig(f"./predict_vis/{i}.png")
        plt.close()

        if (i > 20):
            break
if __name__ == "__main__":
    plot_prenms()