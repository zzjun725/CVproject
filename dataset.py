import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rpn import RPNHead
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, paths, indices=None,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Resize(size=(800, 1066)),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                     transforms.Pad(padding=(11, 0), fill=0)
                 ])):
        img, mask, bbox, label = paths
        self.imageScale = 1066 / 400
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(800, 1066)),
            transforms.Pad(padding=(11, 0), fill=0)
        ])
        with h5py.File(img, 'r') as img:
            # shape=(3265, 3, 300, 400), type=ndarray
            self.image = img['data'][:]
        with h5py.File(mask, 'r') as mask:
            # shape=(3843, 300, 400)
            self.mask = mask['data'][:]
        # shape=(3265, ...), ... as(n, 4)
        self.bbox = np.load(bbox, allow_pickle=True)
        # shape=(3265, ...), ... as(m, )
        self.label = np.load(label, allow_pickle=True)
        if indices is None:
            self.indices = np.random.permutation(len(self.image))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label_sum = 0
        idx = self.indices[idx]
        for label in self.label[:idx]:
            label_sum += len(label)
        labels = len(self.label[idx])
        masks = self.mask[label_sum:label_sum + labels].astype(np.uint8)
        image = self.transform(self.image[idx].astype(np.uint8).transpose(1, 2, 0))
        for i, mask in enumerate(masks):
            if i == 0:
                res_masks = self.mask_transform(mask)
            else:
                res_masks = torch.vstack([res_masks, self.mask_transform(mask)])
        bbox = self.bbox[idx] * self.imageScale
        # xywh style bounding box
        bbox = np.stack([bbox[:, 0]/2 + bbox[:, 2]/2, bbox[:, 1]/2 + bbox[:, 3]/2, bbox[:, 2]-bbox[:, 0], bbox[:, 3]-bbox[:, 1]], axis=-1)
        bbox = torch.tensor(bbox)
        label = torch.tensor(self.label[idx], requires_grad=False)

        return image, label, res_masks, bbox, idx


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        images, labels, masks, bounding_boxes, idx = list(zip(*batch))
        data_batch = {"img": torch.stack(images), "bbox": bounding_boxes, "labels": labels, "masks": masks, "idx": torch.tensor(idx)}
        return data_batch

    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, bboxes_path, labels_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = test_build_loader.loader()

    for i, batch in enumerate(train_loader, 0):
        images = batch['img'][0, :, :, :]
        indexes = batch['idx']
        boxes = batch['bbox']
        gt, ground_coord = rpn_net.create_batch_truth(boxes, indexes, images.shape[-2:])

        # Flatten the ground truth and the anchors
        flatten_coord, flatten_gt, flatten_anchors = output_flattening(ground_coord, gt, rpn_net.get_anchors())

        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord = output_decoding(flatten_coord, flatten_anchors)

        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        images = transforms.functional.normalize(images,
                                                 [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(images.permute(1, 2, 0))

        find_cor = (flatten_gt == 1).nonzero()
        find_neg = (flatten_gt == -1).nonzero()

        for elem in find_cor:
            coord = decoded_coord[elem, :].view(-1)
            anchor = flatten_anchors[elem, :].view(-1)

            col = 'r'
            rect = patches.Rectangle((coord[0] - coord[2] / 2, coord[1] - coord[3] / 2), coord[2], coord[3], fill=False,
                                     color=col)
            ax.add_patch(rect)
            rect = patches.Rectangle((anchor[0] - anchor[2] / 2, anchor[1] - anchor[3] / 2), anchor[2], anchor[3],
                                     fill=False, color='b')
            ax.add_patch(rect)
        
        # for elem in find_neg:
        #     anchor = flatten_anchors[elem, :].view(-1)

        #     rect = patches.Rectangle((anchor[0] - anchor[2] / 2, anchor[1] - anchor[3] / 2), anchor[2], anchor[3],
        #                              fill=False, color='g')
        #     ax[1].add_patch(rect)

        plt.savefig(f"./gt_vis/{i}.png")
        plt.close()

        if (i > 20):
            break
