"""
KeypointNet dataset

Author: Matteo Bastico & Pierre Onghena
Please cite our work if the code is helpful to you.
"""
import os
import json
import torch
import numpy as np

import pointops
from pointcept.utils.logger import get_root_logger
# import pointops
from .builder import DATASETS
from .defaults import DefaultDataset
from .utils import naive_read_pcd
from .utils import load_geodesics
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances_argmin
from .utils import closest_distance_with_batch


@DATASETS.register_module()
class KeyboneNetDataset(DefaultDataset):
    def __init__(
            self,
            data_root="data/KeyboneNetPlus",
            category="FB",
            num_points=2048,
            uniform_sampling=True,
            save_record=True,
            pad_index=-1,
            max_keypoints=None,
            cls_token_shift=None,
            **kwargs
    ):
        """
        kwargs:
            split="train",
            data_root="data/dataset",
            transform=None,
            test_mode=False,
            test_cfg=None,
            cache=False,
            cache=False,
            ignore_index=-1,
            loop=1,
        """
        self.data_root = data_root
        self.category = category.lower()
        self.num_points = num_points
        self.uniform_sampling = uniform_sampling
        self.pad_index = pad_index
        self.cls_token_shift = cls_token_shift
        self.categories = []
        self.token2category = {}
        with open(os.path.join(self.data_root, "synsetoffset2category.txt"), "r") as f:
            for line in f:
                ls = line.strip().split()
                self.token2category[ls[1]] = len(self.categories)
                self.categories.append(ls[0].lower())
        self.category2token = {v: k for k, v in self.token2category.items()}

        super().__init__(data_root=self.data_root, **kwargs)
        self.keypoints = self.get_keypoints()
        # self.geodesics = self.get_geodesics()
        self.max_keypoints = max(max(kp[1] for kp in keypoint) for keypoint in self.keypoints.values()) + 1 if max_keypoints is None else max_keypoints

        # check, prepare record
        record_name = f"keypointnet_{self.split}_{self.category}"
        if num_points is not None:
            record_name += f"_{num_points}points"
            if uniform_sampling:
                record_name += "_uniform"
        record_path = os.path.join(self.data_root, f"{record_name}.pth")
        logger = get_root_logger()
        if os.path.isfile(record_path):
            logger.info(f"Loading record: {record_name} ...")
            self.data = torch.load(record_path, weights_only=False)
        else:
            logger.info(f"Preparing record: {record_name} ...")
            self.data = {}
            for idx in range(len(self.data_list)):
                data_name = self.data_list[idx]
                logger.info(f"Parsing data [{idx}/{len(self.data_list)}]: {data_name}")
                self.data[data_name] = self.get_data(idx)
            if save_record:
                torch.save(self.data, record_path)

    def get_data_list(self):
        splits_path = os.path.join(self.data_root, 'splits')
        split_path = os.path.join(splits_path, "{}.txt".format(self.split))
        data_list = np.loadtxt(split_path, dtype="str")
        # If there is just one sample
        if len(data_list.shape) == 0:
            data_list = np.expand_dims(data_list, 0)
        if self.category == 'all':
            return data_list
        else:
            data_list = [entry for entry in data_list if entry.startswith(
                self.category2token[self.categories.index(self.category)]
            )]
            return data_list

    def get_keypoints(self):
        annots = json.load(open(os.path.join(self.data_root, "annotations", self.category + '.json')))
        keypoints = dict([(annot['class_id'] + annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id'])
                                               for kp_info in annot['keypoints']]) for annot in annots])
        return keypoints

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]
        if data_name in self.data.keys():
            return self.data[data_name]
        else:
            # Separate category and name
            class_id = data_name.split('-', 1)[0]
            model_id = data_name.split('-', 1)[-1].rstrip('\n')
            data_path = os.path.join(
                self.data_root, "pcds", class_id, model_id + ".pcd"
            )
            coord = naive_read_pcd(data_path, color=False)
            keypoints, semantic_id = map(list, zip(*self.keypoints[class_id + model_id]))
            coord_keypoints = coord[keypoints]

            # Sampling
            if self.num_points is not None:
                if self.uniform_sampling:
                    with torch.no_grad():
                        mask = pointops.farthest_point_sampling(
                            torch.tensor(coord).float().cuda(),
                            torch.tensor([len(coord)]).long().cuda(),
                            torch.tensor([self.num_points]).long().cuda(),
                        )
                    coord = coord[mask.cpu()]
                else:
                    np.random.shuffle(coord)
                    coord = coord[: self.num_points]

            kdtree = KDTree(coord)
            for idx, coord_keypoint in enumerate(coord_keypoints):
                keypoints[idx] = kdtree.query([coord_keypoint])[1].item()

            label = -np.ones((self.max_keypoints,), dtype=np.int64)
            label[semantic_id] = keypoints

            # From: https://github.com/qq456cvb/KeypointNet/blob/master/benchmark_scripts/dataset.py
            labels = [label]

            cls_token = self.token2category[
                os.path.basename(os.path.dirname(data_path))
            ]
            if self.cls_token_shift is not None:
                cls_token += self.cls_token_shift

            data_dict = dict(
                coord=coord,
                kp_index=[np.concatenate(labels)],  # Workaround to generate offset for kp_index
                cls_token=cls_token,
                name=data_name.rstrip('\n')
            )
            return data_dict

    def prepare_test_data(self, idx):
        # load data
        return self.prepare_train_data(idx)
