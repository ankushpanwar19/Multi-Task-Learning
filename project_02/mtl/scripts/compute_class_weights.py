import sys
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from tqdm import tqdm

sys.path.append('/Users/abhinavaggarwal/Downloads/dlad_project/project_02/')
from mtl.datasets.definitions import SPLIT_TRAIN, MOD_SEMSEG
from mtl.utils.helpers import resolve_dataset_class
from mtl.utils.transforms import get_transforms


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Expecting 2 arguments: dataset name and path')
        exit(0)

    dataset_name, dataset_root = sys.argv[1], sys.argv[2]
    print(f'Computing class wights of {dataset_name}')

    dataset_cls = resolve_dataset_class(dataset_name)
    ds = dataset_cls(dataset_root, SPLIT_TRAIN)

    transforms = get_transforms()
    ds.set_transforms(transforms)

    dl = DataLoader(ds, 64, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    
    weights = np.zeros(19)

    for batch in tqdm(dl):
        seg = batch[MOD_SEMSEG]

        weights += np.bincount(batch["semseg"].numpy().flatten())[:19]*1e-10


    weights_inv = 1/weights
    weights_inv_sum = np.sum(weights_inv)
    final_weights = weights_inv/weights_inv_sum

    class_names = [clsdesc.name for clsdesc in Cityscapes.classes if not clsdesc.ignore_in_eval]

    for i, weight in enumerate(final_weights):
        print(f'{class_names[i]}: {weight}')
