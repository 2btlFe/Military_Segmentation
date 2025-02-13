import numpy as np
import os
import imageio
import torch
import cv2

from utils import load_config, create_output_dir_and_save_config
from dataset import cityscale_data_partition, read_rgb_img, get_patch_info_one_img
from dataset import spacenet_data_partition
from sam_road import SAMRoad
# from triage import visualize_image_and_graph, rasterize_graph
import pickle
import scipy
from collections import defaultdict
import time
import ipdb


def get_img_paths(root_dir, image_indices):
    img_paths = []

    for ind in image_indices:
        img_paths.append(os.path.join(root_dir, f"region_{ind}_sat.png"))

    return img_paths



def crop_img_patch(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1, :]


def get_batch_img_patches(img, batch_patch_info):
    # ipdb.set_trace()
    patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch


def infer_one_img(net, img, config):
    # TODO(congrui): centralize these configs
    # ipdb.set_trace()
    image_size = img.shape[0]

    batch_size = config.INFER_BATCH_SIZE    # batch 하나가 적용할 샘플의 개수 (64) 512 / 64 = 16 batches  
    # list of (i, (x_begin, y_begin), (x_end, y_end))
    all_patch_info = get_patch_info_one_img(
        0, image_size, config.SAMPLE_MARGIN, config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE)
    # 1024 x 1024 기준 (16 x 16) 개의 patch를 만든다    

    patch_num = len(all_patch_info)
    batch_num = (
        patch_num // batch_size
        if patch_num % batch_size == 0
        else patch_num // batch_size + 1
    )

    # ipdb.set_trace()
    device = net.device
    # [IMG_H, IMG_W]
    # fused_keypoint_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(device, non_blocking=False)
    fused_road_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(device, non_blocking=False)
    pixel_counter = torch.zeros(img.shape[0:2], dtype=torch.float32).to(device, non_blocking=False)

    # stores img embeddings for toponet
    # list of [B, D, h, w], len=batch_num
    # img_features = list()

    for batch_index in range(batch_num):
        start_time = time.time()
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]
        # tensor [B, H, W, C]
        batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        with torch.no_grad():
            batch_img_patches = batch_img_patches.to(device, non_blocking=False)
            # [B, H, W, 2]
            mask_scores, patch_img_features = net.infer_masks_and_img_features(batch_img_patches)
            # img_features.append(patch_img_features)
        # Aggregate masks
        # batch size - 64개에 대해서 적용
         
        for patch_index, patch_info in enumerate(batch_patch_info):
            _, (x0, y0), (x1, y1) = patch_info
            road_patch = mask_scores[patch_index, :, :, 1]
            # fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
            fused_road_mask[y0:y1, x0:x1] += road_patch
            pixel_counter[y0:y1, x0:x1] += torch.ones(road_patch.shape[0:2], dtype=torch.float32, device=device)
     
        # for patch_index, patch_info in enumerate(batch_patch_info):
        #     _, (x0, y0), (x1, y1) = patch_info
        #     keypoint_patch, road_patch = mask_scores[patch_index, :, :, 0], mask_scores[patch_index, :, :, 1]
        #     fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
        #     fused_road_mask[y0:y1, x0:x1] += road_patch
        #     pixel_counter[y0:y1, x0:x1] += torch.ones(road_patch.shape[0:2], dtype=torch.float32, device=device)

        print(f"road_sam - sliding_window_{batch_index}", time.time() - start_time)

    # fused_keypoint_mask /= pixel_counter
    fused_road_mask /= pixel_counter
    # range 0-1 -> 0-255
    # fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
    fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()

    return fused_road_mask

    
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
