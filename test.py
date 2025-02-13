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

# SAM for instance segmentation
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from open_earth_map.utils import sam_refinement

# SAM for road extraction 
from sam_road import SAMRoad
from utils import load_config, create_output_dir_and_save_config
from inference import infer_one_img

warnings.filterwarnings("ignore")

# 마스크 하나에 대해 majority voting하기 
def process_single_mask(args):
                msk, mask = args
                binary_mask = msk['segmentation']
                mask_res = np.zeros_like(binary_mask)
                mask_extracted = mask[binary_mask]

                # 아예 background인 경우 그냥 background로 처리
                if (mask_extracted != 0).sum() == 0:
                    return mask_res, binary_mask, 0

                mask_extracted = mask_extracted[mask_extracted != 0]
                mask_cls = np.bincount(mask_extracted.flatten()).argmax()

                # background제거 하고 majority voting하기
                mask_res[binary_mask] = mask_cls
                
                return mask_res, binary_mask, mask_cls

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
    # ipdb.set_trace()

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
    Args.add_argument("--crf", action="store_true")
    Args.add_argument("--fig_dir", type=str, default="skku_predictions")
    Args.add_argument("--sam_mask", action="store_true")
    Args.add_argument("--config", type=str, default="config/toponet_vitb_512_cityscale_viz.yaml")
    Args.add_argument("--checkpoint", type=str, default="/workspace/ssd0/byeongcheol/Remote_Sensing/open_earth_map/cityscale_vitb_512_e10.ckpt")
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
    
    # Load Test Dataset
    skku_tile=args.skku_dir
    skku_tile_list = sorted(os.listdir(skku_tile))
    # ipdb.set_trace()
    skku_tile_list = [os.path.join(skku_tile, f) for f in skku_tile_list if f.endswith(".png")]
    test_data = oem.dataset.OpenEarthMapDataset(skku_tile_list, n_classes=args.n_classes, augm=None, testing=True)
    
    # Output Directory
    OUTPUT_DIR=f"{args.model_dir}/{args.fig_dir}"
    # OUTPUT_DIR_SAM=f"{args.model_dir}/{args.fig_dir}_sam"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # os.makedirs(OUTPUT_DIR_SAM, exist_ok=True)
    
    network.eval().to(DEVICE1)
    arguments = []
    arguments_sam = []
    
    if args.sam_mask:
        sam_ckpt = "sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(device=DEVICE1).eval()
        mask_generator = SamAutomaticMaskGenerator(sam)
    

    # device1 print DEVICE1 memory
    # print(torch.cuda.memory_allocated)

    # device2 print DEVICE2 memory
    for idx, filename in tqdm(enumerate(range(len(test_data))), total=len(test_data)):
        img, fn = test_data[idx][0], test_data[idx][2]

        # SAM_Road Inference
        start_time =time.time()
        with torch.no_grad():
            road_mask = infer_one_img(road_net, img.permute(1, 2, 0) * 255.0, config)

            road_mask_max = np.max(road_mask)   #249
            road_mask_min = np.min(road_mask)   # 0
            road_mask = (road_mask - road_mask_min) / (road_mask_max - road_mask_min)

            rg = 0.1            
            # 1) thresholding
            road_mask[road_mask < rg] = 0
            road_mask[road_mask >= rg] = 1 
        road_sam_tmp_time = time.time() - start_time
        road_sam_time += road_sam_tmp_time
        print("road_time: ", road_sam_time)

        # UNetFormer Inference    
        start_time =time.time()
        with torch.no_grad():
            prd = network(img.unsqueeze(0).to(DEVICE1)).squeeze(0).cpu() 
        segmentation_tmp_time = time.time() - start_time
        segmentation_time += segmentation_tmp_time 
        print("segementation_time: ", segmentation_tmp_time)



        # SAM mask generation & Majority Voting
        if args.sam_mask:    
            img_np = (img.numpy().transpose((1, 2, 0)) * 255.0).astype(np.uint8)
            
            sam_start = time.time()
            sam_masks = mask_generator.generate(img_np)    # 입력 텐서 
            sam_end = time.time()
            sam_tmp_time = sam_end - sam_start
            print(f"Sam mask generation time: {sam_tmp_time:.2f}s")
            sam_time += sam_tmp_time

            mask = np.argmax(prd.numpy(), axis=0)
            mask_visual = np.zeros_like(mask)
            mask_res_list = [] 
            
            #TODO:병렬로 최적화 필요 
            mask_start = time.time()

            results = []            
            args_list = [(msk, mask) for msk in sam_masks]
            for arg in args_list:
                res = process_single_mask(arg)
                results.append(res)
            

            # with mp.Pool(processes=mp.cpu_count()) as pool:
                # args_list = [(msk, mask) for msk in sam_masks]
                # results = pool.map(process_single_mask, args_list)  # Majority Voting 적용

            for mask_res, binary_mask, mask_cls in results:
                mask_res_list.append(mask_res)
                mask_visual[binary_mask] = mask_cls
            mask_end = time.time()
            mask_tmp_time = mask_end - mask_start
            print(f"Mask processing time: {mask_end - mask_start:.2f}s")
            masking_time += mask_tmp_time
        
            # visualize instances
            # mask_inst_vis = np.zeros((mask_visual.shape[0], mask_visual.shape[1], 3))
            # for j, inst in enumerate(mask_res_list):
            #     # ipdb.set_trace()
            #     mask_inst_vis[inst != 0] = np.array(colors[j%100])    # 새로운 색상 넣기
            # mask_inst_vis = mask_inst_vis.astype(np.uint8)
            # arguments_sam.append((fn, mask_inst_vis, os.path.join(OUTPUT_DIR_SAM, fn.split("/")[-1])))

            # visualize mask
            prd = oem.utils.make_rgb(mask_visual)
            object_map = ((mask_visual != 0) * (mask_visual != 4)).astype(np.uint8) # 0: background, 4: road

            # mask_visual_dir = 'mask_visual'
            # os.makedirs(mask_visual_dir, exist_ok=True)
            # # ipdb.set_trace()
            # mask_file_name = os.path.basename(fn).split('.')[0]
            # with open(f'{mask_visual_dir}/{mask_file_name}.pkl', 'wb') as f:
            #     object_map = ((mask_visual != 0) * (mask_visual != 4)).astype(np.uint8) # 0: background, 4: road
            #     pickle.dump((object_map, prd), f)

        elif args.crf:
            cls, h, w = prd.shape
            predictions = np.zeros((h, w))
            
            softmax = torch.nn.functional.softmax(prd, dim=0).numpy()
            unary = unary_from_softmax(softmax)
            d = dcrf.DenseCRF2D(w, h, args.n_classes) # width, height, nlabels
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)

            # normalize
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            img = img.numpy().transpose((1, 2, 0))
            img = (img - mean) / std

            rgb_image = img.astype(np.uint8)   # HxWxC
            rgb_image = np.ascontiguousarray(rgb_image) 
            d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=rgb_image, compat=3)
            Q = d.inference(5)
            mask = np.argmax(Q, axis=0)
            predictions = mask.reshape((h, w)) # HxW
            prd = oem.utils.make_rgb(predictions)
        else:
            prd = oem.utils.make_rgb(np.argmax(prd.numpy(), axis=0))

        fout = os.path.join(OUTPUT_DIR, fn.split("/")[-1])

        # ipdb.set_trace() 
        arguments.append((fn, prd, object_map, road_mask, fout, idx))

    # semantic segmentation
    t0 = time.time()
    # img_writer(arguments[0])
    with mp.Pool(processes=mp.cpu_count()) as pool:
        prds = pool.map(img_writer, arguments)  # 병렬 처리 후 결과 리스트 반환 
    
    # mpp.Pool(processes=mp.cpu_count()).map(img_writer, arguments)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Road Sam time: {road_sam_time:.2f}s")
    print(f"Segmentation time: {segmentation_time:.2f}s")
    print(f"Sam mask generation time: {sam_time:.2f}s")
    print(f"Mask processing time: {masking_time:.2f}s")

    #전체 그림 만들기 - prds가 안 된다 
    ROW = 4
    COL = 3    
    total_image = np.zeros((1024 * ROW, 1024 * COL, 3), dtype=np.uint8)

    sorted_prds = sorted(prds, key=lambda x: x[0])
    for idx in range(len(arguments)):
        row = idx // COL
        col = idx % COL
        assert sorted_prds[idx][0] == idx
        total_image[row*1024:(row+1)*1024, col*1024:(col+1)*1024, :] = sorted_prds[idx][1]

    total_image = cv2.cvtColor(total_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'total_image.png'), total_image)
