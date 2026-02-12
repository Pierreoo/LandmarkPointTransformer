"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from scipy.spatial import distance
import scipy.sparse as ss
import torch.nn.functional as F

import os
import pickle
from sklearn import neighbors
from scipy.sparse.csgraph import shortest_path


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        if "img_num" in batch[0].keys():
            max_img_num = max([d["img_num"] for d in batch])
        batch = {
            key: (
                (
                    collate_fn([d[key] for d in batch])
                    if "offset" not in key
                    # offset -> bincount -> concat bincount-> concat offset
                    else torch.cumsum(
                        collate_fn(
                            [d[key].diff(prepend=torch.tensor([0])) for d in batch]
                        ),
                        dim=0,
                    )
                )
                if "correspondence" not in key
                else collate_fn(
                    [
                        F.pad(
                            d[key].permute(0, 2, 1),
                            (0, max_img_num - d[key].shape[1]),
                            value=-1,
                        ).permute(0, 2, 1)
                        for d in batch
                    ]
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        offset_assets = [asset for asset in batch.keys() if "offset" in asset]
        for offset_asset in offset_assets:
            batch[offset_asset] = torch.cat(
                [batch[offset_asset][1:-1:2], batch[offset_asset][-1].unsqueeze(0)],
                dim=0,
            )
        if "img_num" in batch.keys():
            n = batch["img_num"].shape[0]
            num_pairs = n // 2
            len_pairs = num_pairs * 2
            pairs_tensor = batch["img_num"][:len_pairs]

            if num_pairs == 0:
                pass
            else:
                summed_pairs = pairs_tensor.view(-1, 2).sum(dim=1)
                if n % 2 != 0:
                    last_element = batch["img_num"][-1:]
                    result = torch.cat((summed_pairs, last_element))
                else:
                    result = summed_pairs
                batch["img_num"] = result
        correspondence_assets = [
            asset for asset in batch.keys() if "correspondence" in asset
        ]
        for correspondence_asset in correspondence_assets:
            offset = batch["offset"]
            start = 0
            N, v, n = batch[correspondence_asset].shape
            v2 = v * 2
            batch_correspondence_mix = -torch.ones((N, v2, n))
            for i in range(len(offset)):
                if i % 2 == 0:
                    batch_correspondence_mix[start : offset[i], 0:v] = batch[
                        correspondence_asset
                    ][start : offset[i], 0:v]
                if i % 2 != 0:
                    batch_correspondence_mix[start : offset[i], v:] = batch[
                        correspondence_asset
                    ][start : offset[i], 0:v]
                start = offset[i]
            if len(offset) % 2 == 0:
                pass
            else:
                start = 0 if len(offset) == 1 else offset[-2]
                batch_correspondence_mix[start:N, -v:] = batch[correspondence_asset][
                    start:N, -v:
                ]
            batch[correspondence_asset] = batch_correspondence_mix
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))

def naive_read_pcd(path, color=False):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=float)
    if color:
        colors = np.array(data[:, -1], dtype=int)
        colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
        # Stack them
        return np.hstack((pc, colors)).astype(np.float32)
    else:
        return pc


def normalize_pc(pc):
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def closest_distance_with_batch(p1, p2, is_sum=True):
    """
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param is_sum: whehter to return the summed scalar or the separate distances with indices
    :return: the distances from p1 to the closest points in p2
    """
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    min_dist, min_indice = torch.min(dist, dim=2)
    dist_scalar = torch.sum(min_dist)

    if is_sum:
        return dist_scalar
    else:
        return min_dist, min_indice


def _connect_components(adjacency, coord, repair_connections=1):
    # Check for connected components
    n_components, labels = ss.csgraph.connected_components(adjacency, directed=False)
    if n_components == 1:
        return adjacency
    # Faster
    adjacency = adjacency.tolil()
    # Get unique pairs of disconnected components
    for i, j in np.transpose(np.triu_indices(n_components, k=1)):
        nodes_i = np.where(labels == i)[0]
        nodes_j = np.where(labels == j)[0]
        # Compute pairwise distances and get the indices of the closest n pairs
        distances = distance.cdist(coord[nodes_i], coord[nodes_j], 'euclidean')
        closest_pairs = np.unravel_index(np.argsort(distances, axis=None)[:repair_connections], distances.shape)
        # Map the closest pairs to original node indices
        closest_i = nodes_i[closest_pairs[0]]
        closest_j = nodes_j[closest_pairs[1]]
        # Add the distances as weights to the adjacency matrix in both directions
        adjacency[closest_i, closest_j] = distances[closest_pairs]
        adjacency[closest_j, closest_i] = distances[closest_pairs]
    return adjacency.tocsr()


def gen_geo_dists(pc):
    graph = neighbors.kneighbors_graph(pc, 20, mode='distance', include_self=False)
    graph = _connect_components(graph, pc)
    return shortest_path(graph, directed=False)


def geodesics_worker(data_path, norm=False, coord=None):
    if coord is None:
        pcd = naive_read_pcd(data_path)[:, 0:3]
        if norm:
            pcd = normalize_pc(pcd)
    else:
        pcd = coord
    geo_dist = gen_geo_dists(pcd)
    # data_name = os.path.splitext(os.path.basename(data_path))[0]
    # return data_name, geo_dist
    return geo_dist


def load_geodesics(data_list, data_root, norm=False, task='correspondence', coord_list=None):
    norm_path = "nonorm" if not norm else ""

    geo_dists = {}
    for data_name, coord in zip(data_list, coord_list):
        class_id = data_name.split('-')[0]
        model_id = data_name.split('-')[-1].rstrip('\n')
        # os.makedirs(os.path.join(data_root, 'geodist' + '_' + norm_pathnorm_path, class_id), exist_ok=True)
        os.makedirs(os.path.join(data_root, 'geodesics', class_id), exist_ok=True)

        # fn = os.path.join(data_root, 'geodist' + '_' + norm_path, class_id, '{}.pkl'.format(model_id))
        fn = os.path.join(data_root, 'geodesics', class_id, '{}.pkl'.format(model_id))
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                if task == 'correspondence':
                    geo_dists[data_name.rstrip('\n')] = pickle.load(f)
                elif task == 'saliency':
                    geo_dists[model_id] = pickle.load(f)
        else:
            data_path = os.path.join(data_root, 'pcds', class_id, '{}.pcd'.format(model_id))
            geo_dist = geodesics_worker(data_path, norm=norm, coord=coord)
            with open(fn, 'wb') as f:
                pickle.dump(geo_dist, f)
            if task == 'correspondence':
                geo_dists[data_name.rstrip('\n')] = geo_dist
            elif task == 'saliency':
                geo_dists[model_id] = geo_dist
    return geo_dists