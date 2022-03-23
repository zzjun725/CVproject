import numpy as np
import torch
from functools import partial
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function computes the IOU between two set of boxes
# boxA (m, 4)
# boxB (n, 4)
# xaya bool: indicate boxA in [x1 y1 x2 y2] format or not
# xbyb bool: similar to xaya
# return iou (m, n)
def IOU(boxA, boxB, xaya=False, xbyb=False):
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes
    ##################################
    append_boxA = boxA.unsqueeze(1).repeat(1, len(boxB), 1)
    append_boxB = boxB.unsqueeze(0).repeat(len(boxA), 1, 1)
    if xaya:
        x1 = append_boxA[..., 0]
        x2 = append_boxA[..., 2]
        y1 = append_boxA[..., 1]
        y2 = append_boxA[..., 3]
    else:
        x1 = append_boxA[..., 0] - append_boxA[..., 2] / 2
        x2 = append_boxA[..., 0] + append_boxA[..., 2] / 2
        y1 = append_boxA[..., 1] - append_boxA[..., 3] / 2
        y2 = append_boxA[..., 1] + append_boxA[..., 3] / 2

    if xbyb:
        x3 = append_boxB[..., 0]
        x4 = append_boxB[..., 2]
        y3 = append_boxB[..., 1]
        y4 = append_boxB[..., 3]
    else:
        x3 = append_boxB[..., 0] - append_boxB[..., 2] / 2
        x4 = append_boxB[..., 0] + append_boxB[..., 2] / 2
        y3 = append_boxB[..., 1] - append_boxB[..., 3] / 2
        y4 = append_boxB[..., 1] + append_boxB[..., 3] / 2
    iou = torch.clip(torch.min(x4, x2) - torch.max(x1, x3), 0) * torch.clip(torch.min(y4, y2) - torch.max(y1, y3), 0)
    iou = iou / ((x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - iou)
    return iou

'''
    This function calculates the IOU between two single bounding boxes
'''
def single_IOU(boxA, boxB, xaya=False, xbyb=False):
    if xaya:
        x1, y1, x2, y2 = boxA
    else:
        ax1, ay1 = boxA[0], boxA[1]
        w1, h1 = boxA[2], boxA[3]
        x1, x2 = ax1-w1/2, ax1+w1/2
        y1, y2 = ay1-h1/2, ay1+h1/2
    
    if xbyb:
        x3, y3, x4, y4 = boxB
    else:
        ax2, ay2 = boxB[0], boxB[1]
        w2, h2 = boxB[2], boxB[3]
        x3, x4 = ax2-w2/2, ax2+w2/2
        y3, y4 = ay2-h2/2, ay2+h2/2

    iou = torch.clip(min(x4, x2)-max(x1, x3), 0) * torch.clip(min(y4, y2)-max(y1, y3), 0)
    iou = iou / ((x2-x1) * (y2-y1) + (x4-x3) * (y4-y3) - iou)
    return iou


def max_IOU(boxA, boxB, xaya=False, xbyb=False):
    append_boxA = boxA.unsqueeze(1).repeat(1, len(boxB), 1)
    append_boxB = boxB.unsqueeze(0).repeat(len(boxA), 1, 1)
    if xaya:
        x1 = append_boxA[..., 0]
        x2 = append_boxA[..., 2]
        y1 = append_boxA[..., 1]
        y2 = append_boxA[..., 3]
    else:
        x1 = append_boxA[..., 0] - append_boxA[..., 2] / 2
        x2 = append_boxA[..., 0] + append_boxA[..., 2] / 2
        y1 = append_boxA[..., 1] - append_boxA[..., 3] / 2
        y2 = append_boxA[..., 1] + append_boxA[..., 3] / 2

    if xbyb:
        x3 = append_boxB[..., 0]
        x4 = append_boxB[..., 2]
        y3 = append_boxB[..., 1]
        y4 = append_boxB[..., 3]
    else:
        x3 = append_boxB[..., 0] - append_boxB[..., 2] / 2
        x4 = append_boxB[..., 0] + append_boxB[..., 2] / 2
        y3 = append_boxB[..., 1] - append_boxB[..., 3] / 2
        y4 = append_boxB[..., 1] + append_boxB[..., 3] / 2
    iou = torch.clip(torch.min(x4, x2) - torch.max(x1, x3), 0) * torch.clip(torch.min(y4, y2) - torch.max(y1, y3), 0)
    iou = iou / ((x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - iou)
    max_iou, max_iou_idx = torch.max(iou, dim=1)
    return max_iou, max_iou_idx



# This function flattens the output of the network and the corresponding anchors
# in the sense that it concatenate  the outputs and the anchors from all the grid cells from all
# the FPN levels from all the images into 2D matrices
# Each row correspond of the 2D matrices corresponds to a specific grid cell
# Input:
#       out_r: list:len(FPN){(bz,num_anchors*4,grid_size[0],grid_size[1])}
#       out_c: list:len(FPN){(bz,num_anchors*1,grid_size[0],grid_size[1])}
#       anchors: list:len(FPN){(num_anchor, grid_size[0], grid_size[1], 4)}
# Output:
#       flatten_regr: (total_number_of_anchors*bz,4)
#       flatten_clas: (total_number_of_anchors*bz)
#       flatten_anchors: (total_number_of_anchors*bz,4)
def output_flattening(out_r,out_c,anchors):
    #######################################
    # TODO flatten the output tensors and anchors
    #######################################
    bz= len(out_c[0])
    out_r = torch.cat([i.permute(0, 2, 3, 1).reshape(bz, -1, 4) for i in out_r], dim=1)
    out_c = torch.cat([i.permute(0, 2, 3, 1).reshape(bz, -1) for i in out_c], dim=1)
    anchors = torch.cat([i.permute(1, 2, 0, 3).reshape(-1, 4) for i in anchors], dim=0)
    flatten_regr = out_r.view(-1, 4)
    flatten_clas = out_c.view(-1)
    flatten_anchors = anchors.unsqueeze(0).repeat(bz, 1, 1).view(-1, 4)
    return flatten_regr, flatten_clas, flatten_anchors

# This function decodes the output that is given in the encoded format (defined in the handout)
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors,4)
#       flatten_anchors: (total_number_of_anchors,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding(flatten_out,flatten_anchors, device='cpu'):
    #######################################
    # TODO decode the output
    #######################################
    flatten_anchors = flatten_anchors.to(device)
    flatten_out = flatten_out.to(device)
    x = flatten_out[:, 0] * flatten_anchors[:, 2] + flatten_anchors[:, 0]
    y = flatten_out[:, 1] * flatten_anchors[:, 3] + flatten_anchors[:, 1]
    w = flatten_anchors[:, 2] * torch.exp(flatten_out[:, 2])
    h = flatten_anchors[:, 3] * torch.exp(flatten_out[:, 3])
    x1 = x - w / 2
    x2 = x + w / 2
    y1 = y - h / 2
    y2 = y + h / 2
    box = torch.stack([x1, y1, x2, y2], dim=1)
    return box


# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decodingd(regressed_boxes_t, flatten_proposals, device='cpu'):
    x_p = (flatten_proposals[:, 0] + flatten_proposals[:, 2]) / 2
    y_p = (flatten_proposals[:, 1] + flatten_proposals[:, 3]) / 2
    w_p = - flatten_proposals[:, 0] + flatten_proposals[:, 2]
    h_p = - flatten_proposals[:, 1] + flatten_proposals[:, 3]
    x = regressed_boxes_t[:, 0] * w_p + x_p
    y = regressed_boxes_t[:, 1] * h_p + y_p
    w = w_p * torch.exp(regressed_boxes_t[:, 2])
    h = h_p * torch.exp(regressed_boxes_t[:, 3])
    box = torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=1)
    return box

def flat_anchors(anchors):
    new_anchors = []
    for i in range(len(anchors)):
        new_anchors.append(anchors[i].clone().reshape(-1, 4)) #n*g[0]*g[1], 4
    new_anchors = torch.cat(new_anchors, dim=0)
    return new_anchors