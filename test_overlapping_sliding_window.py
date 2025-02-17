import os
import time
import warnings
import numpy as np
import torch
import rasterio
import cv2
import open_earth_map as oem
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
from open_earth_map.losses import JointLoss
import argparse

# wandb
import wandb

# Multiprocessing
import multiprocessing.pool as mpp
import multiprocessing as mp

# tqdm
from tqdm import tqdm

# densecrf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian
from pydensecrf.utils import unary_from_softmax

# utils
import ipdb
import pickle
from PIL import Image

# SAM for instance segmentation
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from open_earth_map.utils import sam_refinement

# SAM for road extraction 
from sam_road import SAMRoad
from utils import load_config, create_output_dir_and_save_config
from inference import infer_one_img

# inference.py
from inference import get_patch_info_one_img
import torch.nn.functional as F
import math

warnings.filterwarnings("ignore")

# 마스크 하나에 대해 majority voting하기 
def process_single_mask(args):
    # ipdb.set_trace()
    msk, mask = args
    binary_mask = msk['segmentation']
    mask_res = np.zeros_like(binary_mask)
    
    try:
        mask_extracted = mask[binary_mask]
    except:
        ipdb.set_trace()

    # 아예 background인 경우 그냥 background로 처리
    if (mask_extracted != 0).sum() == 0:
        return mask_res, binary_mask, 0

    mask_extracted = mask_extracted[mask_extracted != 0]
    mask_cls = np.bincount(mask_extracted.flatten()).argmax()

    # background제거 하고 majority voting하기
    mask_res[binary_mask] = mask_cls
    
    return mask_res, binary_mask, mask_cls


def process_single_mask_with_average_pooling(args):
    # ipdb.set_trace()
    msk, logit = args
    binary_mask = msk['segmentation']

    logit = logit.transpose(1, 2, 0)

    # mask_res = np.zeros_like(binary_mask)
    logit_extracted = logit[binary_mask]

    # mask 내에서 background 무시하는 것도 고려해야 한다
    logit_mean = np.mean(logit_extracted, axis=0)
    # ipdb.set_trace()
    logit[binary_mask == True] = logit_mean
    logit = logit.transpose(2, 0, 1)

    return logit, binary_mask


# Colors for Instance Visualization
colors = [
    [255, 87, 51],     # Hex #FF5733
    [51, 255, 87],     # Hex #33FF57
    [51, 87, 255],     # Hex #3357FF
    [255, 51, 161],    # Hex #FF33A1
    [161, 51, 255],    # Hex #A133FF
    [51, 255, 161],    # Hex #33FFA1
    [255, 134, 51],    # Hex #FF8633
    [51, 255, 134],    # Hex #33FF86
    [134, 51, 255],    # Hex #8633FF
    [255, 51, 134],    # Hex #FF3386
    [51, 134, 255],    # Hex #3386FF
    [134, 255, 51],    # Hex #86FF33
    [255, 183, 51],    # Hex #FFB733
    [183, 255, 51],    # Hex #B7FF33
    [51, 183, 255],    # Hex #33B7FF
    [255, 51, 183],    # Hex #FF33B7
    [51, 255, 183],    # Hex #33FFB7
    [183, 51, 255],    # Hex #B733FF
    [255, 193, 51],    # Hex #FFC133
    [193, 255, 51],    # Hex #C1FF33
    [51, 193, 255],    # Hex #33C1FF
    [255, 51, 193],    # Hex #FF33C1
    [51, 255, 193],    # Hex #33FFC1
    [193, 51, 255],    # Hex #C133FF
    [255, 225, 51],    # Hex #FFE133
    [225, 255, 51],    # Hex #E1FF33
    [51, 225, 255],    # Hex #33E1FF
    [255, 51, 225],    # Hex #FF33E1
    [51, 255, 225],    # Hex #33FFE1
    [225, 51, 255],    # Hex #E133FF
    [255, 240, 51],    # Hex #FFF033
    [240, 255, 51],    # Hex #F0FF33
    [51, 240, 255],    # Hex #33F0FF
    [255, 51, 240],    # Hex #FF33F0
    [51, 255, 51],     # Hex #33FF33
    [51, 51, 255],     # Hex #3333FF
    [255, 51, 51],     # Hex #FF3333
    [51, 255, 255],    # Hex #33FFFF
    [255, 255, 51],    # Hex #FFFF33
    [255, 51, 255],    # Hex #FF33FF
    [102, 51, 153],    # Hex #663399
    [153, 204, 0],     # Hex #99CC00
    [255, 153, 0],     # Hex #FF9900
    [0, 153, 255],     # Hex #0099FF
    [204, 0, 51],      # Hex #CC0033
    [51, 204, 153],    # Hex #33CC99
    [153, 0, 204],     # Hex #9900CC
    [204, 153, 0],     # Hex #CC9900
    [0, 102, 204],     # Hex #0066CC
    [204, 0, 102],     # Hex #CC0066
    [102, 204, 51],    # Hex #66CC33
    [51, 102, 204],    # Hex #3366CC
    [204, 102, 51],    # Hex #CC6633
    [102, 204, 153],   # Hex #66CC99
    [153, 102, 204],   # Hex #9966CC
    [204, 153, 102],   # Hex #CC9966
    [102, 153, 204],   # Hex #6699CC
    [204, 102, 153],   # Hex #CC6699
    [153, 204, 102],   # Hex #99CC66
    [102, 204, 51],    # Hex #66CC33 (중복)
    [51, 153, 204],    # Hex #3399CC
    [204, 51, 153],    # Hex #CC3399
    [153, 51, 204],    # Hex #9933CC
    [51, 204, 102],    # Hex #33CC66
    [204, 51, 255],    # Hex #CC33FF
    [51, 255, 204],    # Hex #33FFCC
    [255, 51, 204],    # Hex #FF33CC
    [51, 204, 255],    # Hex #33CCFF
    [255, 204, 51],    # Hex #FFCC33
    [204, 255, 51],    # Hex #CCFF33
    [102, 51, 255],    # Hex #6633FF
    [51, 255, 102],    # Hex #33FF66
    [255, 102, 51],    # Hex #FF6633
    [102, 255, 51],    # Hex #66FF33
    [51, 102, 255],    # Hex #3366FF
    [255, 51, 102],    # Hex #FF3366
    [102, 255, 204],   # Hex #66FFCC
    [204, 255, 102],   # Hex #CCFF66
    [102, 204, 255],   # Hex #66CCFF
    [255, 102, 204],   # Hex #FF66CC
    [204, 102, 255],   # Hex #CC66FF
    [102, 204, 255],   # Hex #66CCFF 
    [204, 102, 204],   # Hex #CC66CC
    [102, 204, 204],   # Hex #66CCCC
    [204, 204, 102],   # Hex #CCCC66
    [204, 255, 204],   # Hex #CCFFCC
    [255, 204, 204],   # Hex #FFCCCC
    [204, 255, 239],   # Hex #CCFFEF
    [239, 255, 204],   # Hex #EFFFCC
    [252, 206, 255],   # Hex #FCCEFF
    [207, 239, 252],   # Hex #CFEFFC
    [255, 239, 207],   # Hex #FFEFCF
    [207, 255, 238],   # Hex #CFFFEE
    [255, 238, 255],   # Hex #FFEEFF
    [238, 255, 238],   # Hex #EEFFEE
    [238, 238, 255],   # Hex #EEEEFF
    [255, 238, 238],   # Hex #FFEEEE
    [238, 255, 255],   # Hex #EEFFFF
    [255, 255, 238],   # Hex #FFFFEE
    [254, 220, 186]    # Hex #FEDCBA
]

def img_writer(arguments):
    fn, prd, object_map, road_mask, fout, idx = arguments

    # road_mask만 출력하기
    fout_dir = os.path.join(os.path.dirname(fout), 'road')
    os.makedirs(fout_dir, exist_ok=True)
    fout_filename = os.path.basename(fout)

    road_fout = os.path.join(fout_dir, fout_filename)
    cv2.imwrite(road_fout, road_mask*255)

    # prd와 road_mask 합치기
    road_mask = road_mask // 1
    difference = road_mask * (1 - object_map)  
    prd[difference == 1] = [255, 255, 0]
    
    rgb_mask = cv2.cvtColor(prd, cv2.COLOR_RGB2BGR)
    rescale = 0.3
    fout.replace(".tif", "_sam.png")
    cv2.resize(rgb_mask, (int(rgb_mask.shape[1]*rescale), int(rgb_mask.shape[0]*rescale)), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(fout, rgb_mask)

    return (idx, prd)
    # with rasterio.open(fn, "r") as src:
    #     profile = src.profile
    #     prd = cv2.resize(
    #         prd,
    #         (profile["width"], profile["height"]),
    #         interpolation=cv2.INTER_NEAREST,
    #     )
    #     with rasterio.open(fout, "w", **profile) as dst:
    #         for idx in src.indexes:
    #             dst.write(prd[:, :, idx - 1], idx)

if __name__ == "__main__":
    Args = argparse.ArgumentParser()
    Args.add_argument("--data_dir", type=str, default="/workspace/hdd0/byeongcheol/Remote_Sensing/OpenEarthMap_wo_xBD/")
    Args.add_argument("--n_classes", type=int, default=9)
    Args.add_argument("--model_name", type=str, default="model.pth")
    Args.add_argument("--model_dir", type=str, default="outputs")
    Args.add_argument("--skku_dir", type=str, default="/workspace/hdd0/byeongcheol/Remote_Sensing/skku_new_tiles")
    Args.add_argument("--map_image", type=str, default="/workspace/hdd0/byeongcheol/Remote_Sensing/202403409C02020039.tif")
    Args.add_argument("--crf", action="store_true")
    Args.add_argument("--fig_dir", type=str, default="skku_predictions")
    Args.add_argument("--sam_mask", action="store_true")
    Args.add_argument("--config", type=str, default="config/toponet_vitb_512_cityscale_viz.yaml")
    Args.add_argument("--checkpoint", type=str, default="/workspace/ssd0/byeongcheol/Remote_Sensing/open_earth_map/cityscale_vitb_512_e10.ckpt")
    Args.add_argument("--sam_ckpt", type=str, default="sam_vit_h_4b8939.pth")
    Args.add_argument("--patch_size", type=int, default=512)
    Args.add_argument("--sam_crop_size", type=int, default=1024)
    Args.add_argument("--sam_slide_size", type=int, default=512)
    args = Args.parse_args()

    start_time = time.time()
    road_sam_time = 0
    sam_time = 0        # SAM mask를 얻는 과정
    masking_time = 0    # SAM결과랑 UnetFormer 결과랑 합치는 과정 
    segmentation_time = 0

    # Load the model
    network = oem.networks.UNetFormer(n_classes=args.n_classes)
    network = oem.utils.load_checkpoint(network, model_name=args.model_name, model_dir=args.model_dir)

    # 사용 가능한 GPU가 2개 이상일 때만 할당
    if torch.cuda.device_count() >= 2:
        DEVICE1 = torch.device("cuda:0")  # 첫 번째 GPU
        DEVICE2 = torch.device("cuda:1")  # 두 번째 GPU
    else:
        DEVICE1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DEVICE2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"DEVICE1: {DEVICE1}")
    print(f"DEVICE2: {DEVICE2}")    
    network.eval().to(DEVICE1)

    # TODO: Load Road Extraction Model
    config = load_config(args.config)
    road_net = SAMRoad(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    road_net.load_state_dict(checkpoint["state_dict"])
    road_net.eval().to(DEVICE2)
    
    # Load test dataset
    skku_imgs_dir = args.skku_dir
    skku_tile_list = sorted(os.listdir(skku_imgs_dir))

    # ipdb.set_trace()
    # skku_tile_list = [os.path.join(skku_imgs_dir, f) for f in skku_tile_list if f.endswith(".png")]
    skku_tile_list = [os.path.join(skku_imgs_dir, f) for f in skku_tile_list if f.endswith(".tif")]
    test_data = oem.dataset.SKKUDataset(skku_tile_list, n_classes=args.n_classes, augm=None, testing=True, patch_size=args.patch_size)     
    # test_data = oem.dataset.OpenEarthMapDataset(skku_tile_list, n_classes=args.n_classes, augm=None, testing=True)     
    
    # Output Directory
    OUTPUT_DIR=f"{args.model_dir}/{args.fig_dir}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # os.makedirs(OUTPUT_DIR_SAM, exist_ok=True)
    
    network.eval().to(DEVICE1)
    arguments = []
    arguments_sam = []
    
    if args.sam_mask:
        sam_ckpt = args.sam_ckpt
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device=DEVICE1).eval()
        mask_generator = SamAutomaticMaskGenerator(sam)
    
    # device2 print DEVICE2 memory
    for idx, filename in tqdm(enumerate(range(len(test_data))), total=len(test_data)):
        img_ori, fn = test_data[idx]

        # num patch of each edge
        ori_height = img_ori.shape[1]
        ori_width = img_ori.shape[2]
        patch_size = args.patch_size

        num_crop_height = ori_height // patch_size
        num_crop_width = ori_width // patch_size

        num_patch_of_height = num_crop_height * 2 - 1
        num_patch_of_width = num_crop_width * 2 - 1

        # ipdb.set_trace()

        res_img = torch.zeros((args.n_classes, ori_height, ori_width), dtype=torch.float32, device=DEVICE1)
        res_road_mask = np.zeros((ori_height, ori_width), dtype=np.float32)
        pixel_counter = torch.zeros(img_ori.shape[1:3], dtype=torch.float32, device=DEVICE1)
        road_pixel_counter = torch.zeros(img_ori.shape[1:3], dtype=torch.float32, device=DEVICE2)

        # sliding window patch 설정하기------------------------------------------------ 
        patch_info = []
        sample_min = 0
        sample_height_max = img_ori.shape[1] - (patch_size + sample_min)
        sample_width_max = img_ori.shape[2] - (patch_size + sample_min)

        eval_height_samples = np.linspace(start=sample_min, stop=sample_height_max, num=num_patch_of_height)
        eval_width_samples = np.linspace(start=sample_min, stop=sample_width_max, num=num_patch_of_width)

        eval_height_samples = [round(x) for x in eval_height_samples]
        eval_width_samples = [round(x) for x in eval_width_samples]

        for y in tqdm(eval_height_samples, desc="Height", total=len(eval_height_samples)):
            for x in tqdm(eval_width_samples, desc="Width", leave=False, total=len(eval_width_samples)): 
                x0, y0 = x, y
                x1, y1 = x + patch_size, y + patch_size
                img = img_ori[:, y0:y1, x0:x1] 

                # ipdb.set_trace()
                # 1. SAM_Road Inference
                start_time =time.time()
                with torch.no_grad():
                    road_mask = infer_one_img(road_net, img.permute(1, 2, 0) * 255.0, config)
                    # ipdb.set_trace()
                    cv2.imwrite(f"{OUTPUT_DIR}/road/road_mask_{y}_{x}.png", road_mask)

                    res_road_mask[y0:y1, x0:x1] = road_mask

                road_sam_tmp_time = time.time() - start_time
                road_sam_time += road_sam_tmp_time
                print("road_time: ", road_sam_time)

                # 2. UNetFormer Inference    
                start_time =time.time()
                with torch.no_grad():
                    prd = network(img.unsqueeze(0).to(DEVICE1)).squeeze(0)
                segmentation_tmp_time = time.time() - start_time
                segmentation_time += segmentation_tmp_time 
                print("segementation_time: ", segmentation_tmp_time)

                # 중간 출력
                seg_inter_rgb = np.argmax(prd.cpu().numpy(), axis=0)
                seg_inter_rgb = oem.utils.make_rgb(seg_inter_rgb)
                seg_inter_bgr = cv2.cvtColor(seg_inter_rgb, cv2.COLOR_RGB2BGR)
                os.makedirs("seg_inter", exist_ok=True)
                cv2.imwrite(f"seg_inter/seg_inter_{y}_{x}.png", seg_inter_bgr)

                crop_margin = 100

                road_pixel_counter[y0:y1, x0:x1] += torch.ones(prd.shape[1:3], dtype=torch.float32, device=DEVICE2)                
                if not (y0 == 0 or y1 == img_ori.shape[1] or x0 == 0 or x1 == img_ori.shape[2]):
                    y0 = y0 + crop_margin
                    y1 = y1 - crop_margin
                    x0 = x0 + crop_margin
                    x1 = x1 - crop_margin
                    prd = prd[:, crop_margin: patch_size-crop_margin, crop_margin: patch_size-crop_margin]                  

                # y0 = max(0, y0 - crop_margin)
                # y1 = (ori_height, y1 - crop_margin)
                # x0 = max(0, x0 - crop_margin)
                # x1 = min(ori_width, x1 + crop_margin)

                res_img[:, y0:y1, x0:x1] += prd
                pixel_counter[y0:y1, x0:x1] += torch.ones(prd.shape[1:3], dtype=torch.float32, device=DEVICE1)

        prd = (res_img / pixel_counter).cpu().numpy()
        res_road_mask = (res_road_mask / road_pixel_counter.cpu().numpy())
        
        res_road_mask_max = np.max(res_road_mask)   
        res_road_mask_min = np.min(res_road_mask)
        res_road_mask = (res_road_mask - res_road_mask_min) / (res_road_mask_max - res_road_mask_min)

        rg = 0.1
        # 1) thresholding
        res_road_mask[res_road_mask < rg] = 0
        res_road_mask[res_road_mask >= rg] = 1        
        fout = os.path.join(OUTPUT_DIR, fn.split("/")[-1])
        

        prd_rgb = np.zeros((prd.shape[1], prd.shape[2], 3), dtype=np.uint8)
        res_object_map = np.zeros((prd.shape[1], prd.shape[2]), dtype=np.uint8)
        
        # segmentation 결과 출력
        seg_rgb = np.argmax(prd, axis=0)
        seg_rgb = oem.utils.make_rgb(seg_rgb)
        seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fout.replace(".tif", "_seg.png"), seg_bgr)        
        cv2.imwrite(fout.replace(".tif", "_road.png"), res_road_mask*255)
        
        #TODO: SAM Mask majority voting 수정
        # SAM Mask majority voting
        # 3. SAM mask generation & Majority Voting
        crop_size = patch_size

        # # 크게 가져가면 어떻게 되는지 보자
        # sam_crop_size = args.sam_crop_size
        # sam_slide_size = args.sam_slide_size
        # num_crop_height = math.ceil(ori_height / sam_slide_size)
        # num_crop_width = math.ceil(ori_width / sam_slide_size)

        # ipdb.set_trace()
        logit = np.zeros((args.n_classes, ori_height, ori_width), dtype=np.float32)
        pixel_counter = np.zeros((ori_height, ori_width), dtype=np.float32)

        # hard coding
        sam_crop_size_list = [512, 768, 1024]
        sam_slide_size_list = [256, 384, 512]
        for sam_crop_size, sam_slide_size in zip(sam_crop_size_list, sam_slide_size_list):
            num_crop_height = math.ceil(ori_height / sam_slide_size)
            num_crop_width = math.ceil(ori_width / sam_slide_size)         

            # sam_mask_total = np.zeros(prd.shape[1:3], dtype=np.uint8)
            for h_idx in tqdm(range(num_crop_height), desc="Height", total=num_crop_height):
                for w_idx in tqdm(range(num_crop_width), desc="Width", leave=False, total=num_crop_width):
                    y0 = h_idx * sam_slide_size
                    y1 = min(ori_height, y0 + sam_crop_size)
                    x0 = w_idx * sam_slide_size
                    x1 = min(ori_width, x0 + sam_crop_size)

                    img = img_ori[:, y0:y1, x0:x1]
                    img_np = (img.numpy().transpose((1, 2, 0)) * 255.0).astype(np.uint8)
                    
                    sam_start = time.time()
                    sam_masks = mask_generator.generate(img_np)    # 입력 텐서 
                    sam_end = time.time()
                    sam_tmp_time = sam_end - sam_start
                    print(f"Sam mask generation time: {sam_tmp_time:.2f}s")
                    sam_time += sam_tmp_time

                    # mask = np.argmax(prd[:, y0:y1, x0:x1], axis=0)
                    
                    mask_start = time.time()

                    # logit 적용 
                    # ipdb.set_trace()
                    args_list = [(msk, prd[:, y0:y1, x0:x1]) for msk in sam_masks]

                    # sam_mask = np.zeros_like(mask)
                    # for msk in sam_masks:
                    #     sam_mask += msk['segmentation']
                    # sam_mask_total[y0:y1, x0:x1] = sam_mask

                    # os.makedirs(f"{OUTPUT_DIR}/sam", exist_ok=True)
                    # cv2.imwrite(f"{OUTPUT_DIR}/sam/sam_mask_{y0}_{x0}.png", sam_mask * 255)

                    # Mask Average pooling 적용 
                    # ipdb.set_trace()
                    for arg in args_list:
                        # res = process_single_mask(arg)
                        logit_res, binary_mask = process_single_mask_with_average_pooling(arg)
                        # TODO: road는 나중으로 미루기                    
                        # road는 빼기
                        # if mask_cls == 4:
                            # continue
                        logit[:, y0:y1, x0:x1] += logit_res
                        pixel_counter[y0:y1, x0:x1] += binary_mask
                    
                    mask_end = time.time()
                    mask_tmp_time = mask_end - mask_start
                    print(f"Mask processing time: {mask_end - mask_start:.2f}s")
                    masking_time += mask_tmp_time

                    # visualize mask                
                    # prd_rgb[y0:y1, x0:x1] = oem.utils.make_rgb(mask_visual)
                    # object_map = ((mask_visual != 0) * (mask_visual != 4)).astype(np.uint8) # 0: background, 4: road
                    # res_object_map[y0:y1, x0:x1] = object_map

                    # prd_bgr = cv2.cvtColor(prd_rgb, cv2.COLOR_RGB2BGR)
                    # rescale = 0.3
                    # width, height = prd_bgr.shape[1], prd_bgr.shape[0]
                    # cv2.resize(prd_bgr, (int(width*rescale), int(height*rescale)), interpolation=cv2.INTER_NEAREST)
                    # cv2.imwrite(fout.replace(".tif", f"_sam_{y0}_{x0}.png"), prd_bgr)
            
        logit = logit / pixel_counter
        logit = np.argmax(logit, axis=0)
        logit_rgb = oem.utils.make_rgb(logit)        
        res_object_map = ((logit != 0) * (logit != 4)).astype(np.uint8) # 0: background, 4: road
        arguments.append((fn, logit_rgb, res_object_map, res_road_mask, fout.replace('.tif', '.png'), idx))

    # semantic segmentation
    # t0 = time.time()
    # rescale = 0.3
    t0 = time.time()
    img_writer(arguments[0])
    t1 = time.time()
    # cv2.resize(sam_mask_total, (int(sam_mask_total.shape[1]*rescale), int(sam_mask_total.shape[0]*rescale)), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite("sam_total.png", sam_mask_total*255)

    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     prds = pool.map(img_writer, arguments)  # 병렬 처리 후 결과 리스트 반환 

    # t1 = time.time()
    img_write_time = t1 - t0
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Road Sam time: {road_sam_time:.2f}s")
    print(f"Segmentation time: {segmentation_time:.2f}s")
    print(f"Sam mask generation time: {sam_time:.2f}s")
    print(f"Mask processing time: {masking_time:.2f}s")
    print('images writing spends: {} s'.format(img_write_time))
    #----------------------------------------------------------------------------------


    # num_crop_height = 10
    # num_crop_width = 7
    # crop_size = 512
    # rescale = 0.3

    # # skku_tile 
    # img = test_data[0][0].numpy()
   
    # # # 세로 방향 그리드 라인
    # line_thickness = 3  # 노란 선 두께 지정

    # # 세로 방향 그리드 라인
    # for h_idx in tqdm(range(num_crop_height), desc="Height", total=num_crop_height):
    #     img[0, h_idx*crop_size:h_idx*crop_size+line_thickness, :] = 255  # R
    #     img[1, h_idx*crop_size:h_idx*crop_size+line_thickness, :] = 255  # G
    #     img[2, h_idx*crop_size:h_idx*crop_size+line_thickness, :] = 0    # B

    # # 가로 방향 그리드 라인
    # for w_idx in tqdm(range(num_crop_width), desc="Width", leave=False, total=num_crop_width):
    #     img[0, :, w_idx*crop_size:w_idx*crop_size+line_thickness] = 255  # R
    #     img[1, :, w_idx*crop_size:w_idx*crop_size+line_thickness] = 255  # G
    #     img[2, :, w_idx*crop_size:w_idx*crop_size+line_thickness] = 0    # B


    # img = img.transpose(1, 2, 0) 
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = cv2.resize(img, (int(img.shape[1]*rescale), int(img.shape[0]*rescale)), interpolation=cv2.INTER_NEAREST)
    # img = img * 255.0
    # img = img.astype(np.uint8)
    # cv2.imwrite(f"{OUTPUT_DIR}/skku_tile.png", img)