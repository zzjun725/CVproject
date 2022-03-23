from os import remove
import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import numpy as np

from dataset import *
from utils import *
from rpn import RPNHead
from backbone import Resnet50Backbone


imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
paths = [imgs_path, masks_path, bboxes_path, labels_path]
epoch = 50
batch_size = 4
tolerance = 5

def main():
    dataset = BuildDataset(paths)
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    print("Data Loading")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = test_build_loader.loader()
    print("Training Set Size", len(train_dataset), "Validation Set Size", len(test_dataset))
    print("Creating Model")
    device = "cuda:0"
    rpn_net = RPNHead(device=device)
    # backbone = Resnet50Backbone(device=device)
    backbone = None
    print("Setting Optimizer")
    optimizer = SGD(rpn_net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)

    best_loss = 1000.
    early_stopping = 0.
    best_model = None
    loss_total_train = []
    loss_c_train = []
    loss_r_train = []
    loss_total_val = []
    loss_c_val = []
    loss_r_val = []
    for i in range(epoch):
        print(f"\nEpoch {i} begins")
        print("Train:")
        loss_total, loss_c, loss_r = train(rpn_net, backbone, train_loader, optimizer, i)
        loss_total_train += loss_total
        loss_c_train += loss_c
        loss_r_train += loss_r
        print("Validation")
        loss_total, loss_c, loss_r = val(rpn_net, backbone, test_loader, i, draw=i%5==0)
        loss_total_val += loss_total
        loss_c_val += loss_c
        loss_r_val += loss_r
        val_loss_mean = np.mean(np.array(loss_total))
        print("Epoch {} Validation Loss Mean: {:.4f}".format(i, val_loss_mean))
        if i % 5 == 0:
            torch.save(rpn_net.state_dict(), f"./train_result/rpn/rpn_model_{i}.pth")
        if i > 30:
            if val_loss_mean < best_loss:
                best_loss = val_loss_mean
                early_stopping = 0
                best_model = rpn_net.state_dict()
            else:
                early_stopping += 1
            if early_stopping == tolerance:
                break
    torch.save(best_model, "./train_result/rpn/rpn_best_model.pth")
    np.save("./train_result/rpn/rpn_total_train.npy", np.array(loss_total_train))
    np.save("./train_result/rpn/rpn_c_train.npy", np.array(loss_c_train))
    np.save("./train_result/rpn/rpn_r_train.npy", np.array(loss_r_train))
    np.save("./train_result/rpn/rpn_total_val.npy", np.array(loss_total_val))
    np.save("./train_result/rpn/rpn_c_val.npy", np.array(loss_c_val))
    np.save("./train_result/rpn/rpn_r_val.npy", np.array(loss_r_val))

def train(model: RPNHead, backbone, loader, optimizer, i):
    loss_t = []
    loss_c = []
    loss_r = []
    model.train()
    for idx, data_batch in enumerate(loader):
        img = data_batch['img'].to(model.device)
        if backbone is None:
            logits, bbox_regs = model(img)
        else:
            logits, bbox_regs = model(backbone(img))
        ground_clas, ground_coord = model.create_batch_truth(data_batch["bbox"], data_batch["idx"], img.shape[2:])

        loss, loss1, loss2 = model.compute_loss(logits, bbox_regs, ground_clas, ground_coord)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        loss_t.append(loss.item())
        loss_c.append(loss1.item())
        loss_r.append(loss2.item())
        if idx % 50 == 0:
            print("Epoch {} Batch {}: Total Loss {:.3f} ({:.3f}); Class Loss {:.3f} ({:.3f}); Regressor Loss {:.3f} ({:.3f})".format(i, idx, loss.item(), np.array(loss_t).mean(), 
                                                        loss1.item(), np.array(loss_c).mean(), loss2.item(), np.array(loss_r).mean()))
    return loss_t, loss_c, loss_r

def val(model: RPNHead, backbone, loader, i, draw=True):
    loss_t = []
    loss_c = []
    loss_r = []
    model.eval()
    for idx, data_batch in enumerate(loader):
        img = data_batch['img'].to(model.device)
        if backbone is None:
            logits, bbox_regs = model(img)
        else:
            logits, bbox_regs = model(backbone(img))
        ground_clas, ground_coord = model.create_batch_truth(data_batch["bbox"], data_batch["idx"], img.shape[2:])
        loss, loss1, loss2 = model.compute_loss(logits, bbox_regs, ground_clas, ground_coord, eval=True)
        if draw and idx % 50 == 0:
            plot_NMS(model, data_batch, logits, bbox_regs, i, idx)
        loss_t.append(loss.item())
        loss_c.append(loss1.item())
        loss_r.append(loss2.item())
        if idx % 50 == 0:
            print("Valid Epoch {} Batch {}: Total Loss {:.3f} ({:.3f}); Class Loss {:.3f} ({:.3f}); Regressor Loss {:.3f} ({:.3f})".format(i, idx, loss.item(), np.array(loss_t).mean(), 
                                                        loss1.item(), np.array(loss_c).mean(), loss2.item(), np.array(loss_r).mean()))
    return loss_t, loss_c, loss_r

def plot_NMS(model:RPNHead, data_batch, logits, bbox_regs, epoch, idx):
    logits, bbox_regs = model.postprocess(logits, bbox_regs)
    bbox_regs[0] = bbox_regs[0][:200]
    logits[0] = logits[0][:200]
    images = data_batch['img'][0, :, :, :]
    boxes = data_batch['bbox']
    images = transforms.functional.normalize(images,
                                                 [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(images.permute(1, 2, 0))

    for j in range(len(logits[0])):
        if logits[0][j] > 0.5:
            col = 'b'
            coord = bbox_regs[0][j].cpu().detach().numpy()
            rect = patches.Rectangle((coord[0], coord[1]), coord[2]-coord[0], coord[3]-coord[1], fill=False,
                                    color=col)
            ax.add_patch(rect)
    for j in range(len(boxes[0])):
        col = 'r'
        rect = patches.Rectangle((boxes[0][j, 0] - boxes[0][j, 2] / 2, boxes[0][j, 1] - boxes[0][j, 3] / 2), boxes[0][j, 2], boxes[0][j, 3],
                                    fill=False, color=col)
        ax.add_patch(rect)
    ax.title.set_text("After NMS")
    plt.savefig(f"./predict_vis/rpnhead/{epoch}_{idx}.png")
    plt.close()

if __name__ == "__main__":
    main()