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
import wandb

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    Args = argparse.ArgumentParser()
    Args.add_argument("--data_dir", type=str, default="/workspace/hdd0/byeongcheol/Remote_Sensing/OpenEarthMap_wo_xBD/")
    Args.add_argument("--img_size", type=int, default=512)
    Args.add_argument("--n_classes", type=int, default=9)
    Args.add_argument("--lr", type=float, default=0.0001)
    Args.add_argument("--batch_size", type=int, default=4)
    Args.add_argument("--num_epochs", type=int, default=10)
    args = Args.parse_args()
     
    # wandb
    wandb.init(project="OpenEarthMap", name=f'UnetFormer_img_size_{args.img_size}_nclass_{args.n_classes}_lr_{args.lr}_bs_{args.batch_size}_epochs_{args.num_epochs}')
    
    # Path to the OpenEarthMap directory
    OEM_DATA_DIR = args.data_dir

    # Training and validation file list
    TRAIN_LIST = os.path.join(OEM_DATA_DIR, "train.txt")
    VAL_LIST = os.path.join(OEM_DATA_DIR, "val.txt")

    IMG_SIZE = args.img_size
    N_CLASSES = args.n_classes
    LR = args.lr
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    OUTPUT_DIR = f"outputs/UnetFormer_img_size_{IMG_SIZE}_nclass_{N_CLASSES}_lr_{LR}_bs_{BATCH_SIZE}_epochs_{NUM_EPOCHS}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepares training and validation file lists
    fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
    train_fns = [str(f) for f in fns if f.name in np.loadtxt(TRAIN_LIST, dtype=str)]
    val_fns = [str(f) for f in fns if f.name in np.loadtxt(VAL_LIST, dtype=str)]

    print("Total samples      :", len(fns))
    print("Training samples   :", len(train_fns))
    print("Validation samples :", len(val_fns))

    # Prepares training and validation augmentations
    train_augm = torchvision.transforms.Compose(
    [
        oem.transforms.Rotate(),
        oem.transforms.Crop(IMG_SIZE),
    ],
    )

    val_augm = torchvision.transforms.Compose(
        [
            oem.transforms.Resize(IMG_SIZE),
        ],
    )

    # Defines training and validation datasets and dataloaders
    train_data = oem.dataset.OpenEarthMapDataset(
        train_fns,
        n_classes=N_CLASSES,
        augm=train_augm,
    )

    val_data = oem.dataset.OpenEarthMapDataset(
        val_fns,
        n_classes=N_CLASSES,
        augm=val_augm,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=True,
        drop_last=True,
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        num_workers=10,
        shuffle=False,
    )

    # network = oem.networks.UNet(in_channels=3, n_classes=N_CLASSES)
    network = oem.networks.UNetFormer(n_classes=N_CLASSES)
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    criterion = JointLoss(oem.losses.SoftCrossEntropyLoss(smooth_factor=0.05), oem.losses.DiceLoss(class_weights=0.05))

    # Trains and validates the network
    start = time.time()

    max_score = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch + 1}")
        train_logs = oem.runners.train_epoch(
            model=network,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=DEVICE,
        )

        valid_logs = oem.runners.valid_epoch(
            model=network,
            criterion=criterion,
            dataloader=val_data_loader,
            device=DEVICE,
        )
        
        wandb.log({"Train Loss": train_logs["Loss"], "Train Score": train_logs["Score"], "Valid Loss": valid_logs["Loss"], "Valid Score": valid_logs["Score"]})

        epoch_score = valid_logs["Score"]
        if max_score < epoch_score:
            max_score = epoch_score
            oem.utils.save_model(
                model=network,
                epoch=epoch,
                best_score=max_score,
                model_name="model.pth",
                output_dir=OUTPUT_DIR,
            )
            wandb.save(f"{OUTPUT_DIR}/model.pth")
    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))


