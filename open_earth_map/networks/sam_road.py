import numpy as np
import os
import imageio
import torch
import cv2

from utils import load_config, create_output_dir_and_save_config
from dataset import cityscale_data_partition, read_rgb_img, get_patch_info_one_img
from dataset import spacenet_data_partition
from model import SAMRoad
import graph_extraction
import graph_utils
import triage
# from triage import visualize_image_and_graph, rasterize_graph
import pickle
import scipy
import rtree
from collections import defaultdict
import time

from argparse import ArgumentParser
import ipdb

parser = ArgumentParser()
parser.add_argument(
    "--checkpoint", default=None, help="checkpoint of the model to test."
)
parser.add_argument(
    "--config", default=None, help="model config."
)
parser.add_argument(
    "--output_dir", default=None, help="Name of the output dir, if not specified will use timestamp"
)
parser.add_argument("--device", default="cuda", help="device to use for training")
args = parser.parse_args()


def get_img_paths(root_dir, image_indices):
    img_paths = []

    for ind in image_indices:
        img_paths.append(os.path.join(root_dir, f"region_{ind}_sat.png"))

    return img_paths

def crop_img_patch(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1, :]


def get_batch_img_patches(img, batch_patch_info):
    patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch


def infer_one_img(net, img, config):
    # TODO(congrui): centralize these configs
    image_size = img.shape[0]

    batch_size = config.INFER_BATCH_SIZE
    # list of (i, (x_begin, y_begin), (x_end, y_end))
    all_patch_info = get_patch_info_one_img(
        0, image_size, config.SAMPLE_MARGIN, config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE)
    patch_num = len(all_patch_info)
    batch_num = (
        patch_num // batch_size
        if patch_num % batch_size == 0
        else patch_num // batch_size + 1
    )

    # [IMG_H, IMG_W]
    fused_keypoint_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    fused_road_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    pixel_counter = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)

    # stores img embeddings for toponet
    # list of [B, D, h, w], len=batch_num
    img_features = list()

    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size] # 64개
        # tensor [B, H, W, C]
        batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        with torch.no_grad():
            batch_img_patches = batch_img_patches.to(args.device, non_blocking=False)
            # [B, H, W, 2]

            # 마스크 score 모아서 저장한다
            # 여기서 inference한다 
            mask_scores, patch_img_features = net.infer_masks_and_img_features(batch_img_patches)
            img_features.append(patch_img_features)
        # Aggregate masks
        for patch_index, patch_info in enumerate(batch_patch_info):
            _, (x0, y0), (x1, y1) = patch_info
            keypoint_patch, road_patch = mask_scores[patch_index, :, :, 0], mask_scores[patch_index, :, :, 1]
            fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
            fused_road_mask[y0:y1, x0:x1] += road_patch
            pixel_counter[y0:y1, x0:x1] += torch.ones(road_patch.shape[0:2], dtype=torch.float32, device=args.device)
    
    fused_keypoint_mask /= pixel_counter
    fused_road_mask /= pixel_counter
    # range 0-1 -> 0-255
    fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
    fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()
    
    return fused_keypoint_mask, fused_road_mask

    


if __name__ == "__main__":
    config = load_config(args.config)
    
    # Builds eval model    
    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    net = SAMRoad(config)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print(f'##### Loading Trained CKPT {args.checkpoint} #####')
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(device)

    if config.DATASET == 'cityscale':
        _, _, test_img_indices = cityscale_data_partition()
        rgb_pattern = './cityscale/20cities/region_{}_sat.png'
        gt_graph_pattern = 'cityscale/20cities/region_{}_graph_gt.pickle'
    elif config.DATASET == 'spacenet':
        _, _, test_img_indices = spacenet_data_partition()
        rgb_pattern = './spacenet/RGB_1.0_meter/{}__rgb.png'
        gt_graph_pattern = './spacenet/RGB_1.0_meter/{}__gt_graph.p'
    
    output_dir_prefix = './save/infer_'
    if args.output_dir:
        output_dir = create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=f'./save/{args.output_dir}')
    else:
        output_dir = create_output_dir_and_save_config(output_dir_prefix, config)
    
    total_inference_seconds = 0.0

    for img_id in test_img_indices:
        print(f'Processing {img_id}')
        # [H, W, C] RGB
        img = read_rgb_img(rgb_pattern.format(img_id))
        start_seconds = time.time()
        # coords in (r, c)
        pred_nodes, pred_edges, itsc_mask, road_mask = infer_one_img(net, img, config)
        end_seconds = time.time()
        total_inference_seconds += (end_seconds - start_seconds)

        gt_graph_path = gt_graph_pattern.format(img_id)
        gt_graph = pickle.load(open(gt_graph_path, "rb"))
        gt_nodes, gt_edges = graph_utils.convert_from_sat2graph_format(gt_graph)
        if len(gt_nodes) == 0:
            gt_nodes = np.zeros([0, 2], dtype=np.float32)

        if config.DATASET == 'spacenet':
            # convert ??? -> xy -> rc
            gt_nodes = np.stack([gt_nodes[:, 1], 400 - gt_nodes[:, 0]], axis=1)
            gt_nodes = gt_nodes[:, ::-1]

        # RGB already
        viz_img = np.copy(img)
        img_size = viz_img.shape[0]

        # visualizes fused masks
        mask_save_dir = os.path.join(output_dir, 'mask')
        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_road.png'), road_mask)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_itsc.png'), itsc_mask)

        # # Visualizes the diff between rasterized pred/gt graphs.
        # rast_pred = triage.rasterize_graph(pred_nodes / img_size, pred_edges, img_size, dilation_radius=1)
        # rast_pred_dilate = triage.rasterize_graph(pred_nodes / img_size, pred_edges, img_size, dilation_radius=5)
        # rast_gt = triage.rasterize_graph(gt_nodes / img_size, gt_edges, img_size, dilation_radius=1)
        # rast_gt_dilate = triage.rasterize_graph(gt_nodes / img_size, gt_edges, img_size, dilation_radius=5)

        # fp_pred = (np.less_equal(rast_gt_dilate, 0) * np.greater(rast_pred, 0)).astype(np.uint8)
        # missed_gt = (np.less_equal(rast_pred_dilate, 0) * np.greater(rast_gt, 0)).astype(np.uint8)

        # diff_img = np.array(viz_img)
        # # FP in blue, missed in red (BGR for opencv)
        # diff_img = diff_img * np.less_equal(fp_pred, 0) + fp_pred * np.array([255, 0, 0], dtype=np.uint8)
        # diff_img = diff_img * np.less_equal(missed_gt, 0) + missed_gt * np.array([0, 0, 255], dtype=np.uint8)

        # diff_save_dir = os.path.join(output_dir, 'diff')
        # if not os.path.exists(diff_save_dir):
        #     os.makedirs(diff_save_dir)
        # cv2.imwrite(os.path.join(diff_save_dir, f'{img_id}.png'), diff_img)

        # Visualizes merged large map
        viz_save_dir = os.path.join(output_dir, 'viz')
        if not os.path.exists(viz_save_dir):
            os.makedirs(viz_save_dir)
        viz_img = triage.visualize_image_and_graph(viz_img, pred_nodes / img_size, pred_edges, viz_img.shape[0])
        cv2.imwrite(os.path.join(viz_save_dir, f'{img_id}.png'), viz_img)

        # Saves the large map
        if config.DATASET == 'spacenet':
            # r, c -> ???
            pred_nodes = np.stack([400 - pred_nodes[:, 0], pred_nodes[:, 1]], axis=1)
        large_map_sat2graph_format = graph_utils.convert_to_sat2graph_format(pred_nodes, pred_edges)
        graph_save_dir = os.path.join(output_dir, 'graph')
        if not os.path.exists(graph_save_dir):
            os.makedirs(graph_save_dir)
        graph_save_path = os.path.join(graph_save_dir, f'{img_id}.p')
        with open(graph_save_path, 'wb') as file:
            pickle.dump(large_map_sat2graph_format, file)
        
        print(f'Done for {img_id}.')
    
    # log inference time
    time_txt = f'Inference completed for {args.config} in {total_inference_seconds} seconds.'
    print(time_txt)
    with open(os.path.join(output_dir, 'inference_time.txt'), 'w') as f:
        f.write(time_txt)
