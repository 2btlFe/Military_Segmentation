<div align="center">

![logo](https://github.com/bao18/open_earth_map/blob/main/pics/openearthmap.png)
[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/bao18/open_earth_map/blob/main/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.12+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/previous-versions/) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/) 

</div>

<!-- 
# OpenEarthMap
Quick start in OpenEarthMap  -->
<!-- The main features of this library are:

 - High-level API (only two lines to create a neural network)
 - Three models architectures for multi-class segmentation (including the popular U-Net)
 - Popular metrics and losses for training routines -->

### Overview
OpenEarthMap is a benchmark dataset for global high-resolution land cover mapping. OpenEarthMap consists of 5000 aerial and satellite images with manually annotated 8-class land cover labels and 2.2 million segments at a 0.25-0.5m ground sampling distance, covering 97 regions from 44 countries across 6 continents. OpenEarthMap fosters research including but not limited to semantic segmentation and domain adaptation. Land cover mapping models trained on OpenEarthMap generalize worldwide and can be used as off-the-shelf models in a variety of applications. Project Page: [https://open-earth-map.org/](https://open-earth-map.org/)

### Reference
```
@inproceedings{xia_2023_openearthmap,
    title = {OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping},
    author = {Junshi Xia and Naoto Yokoya and Bruno Adriano and Clifford Broni-Bediako},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month = {January},
    year = {2023}
}
```

### License
<!-- Label data of OpenEarthMap are provided under the same license as the original RGB images, which varies with each source dataset. Label data for regions where the original RGB images are in the public domain or where the license is not explicitly stated are licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) International License. For more details, please see the attribution of source data in the supplementary document of our paper ([https://arxiv.org/abs/2210.10732](https://arxiv.org/abs/2210.10732)).

 -->
Label data of OpenEarthMap are provided under the same license as the original RGB images, which varies with each source dataset. For more details, please see the attribution of source data [here](https://open-earth-map.org/attribution.html). Label data for regions where the original RGB images are in the public domain or where the license is not explicitly stated are licensed under a Creative [Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) International License.

### Setup
```
docker pull 2btlfe/lbc_open_earth_map
```
- 만약 직접 환경 설정을 원할 경우
```
pip install -r requirement.txt
```

### Train (UnetFormer - OpenEarthMap Dataset) 
- 학습 시킨 pth 파일은 ./model/model.pth를 참고하면 된다 -> vclab@115.145.172.165:/mnt/ssd0/byeongcheol/Remote_Sensing/open_earth_map/model/model.pth 에서 다운로드 받을 수 있다
- img_size(768), # of class(9), lr(0.0001), batch size(8), epoch(10) 으로 학습시킨 결과다 
``` python 
bash train.sh
```

### Test (with SAM) 
- skku_dir에 들어갈 이미지들은 vclab@115.145.172.165:/mnt/hdd0/byeongcheol/Remote_Sensing/skku_tiles_40에서 다운로드 할 수 있다
- 이 모델은 SAM을 필요로 한다. "sam_vit_h_4b8939.pth" 을 다운로드 받아서 사용해야 한다. 이 또한, vclab@115.145.172.165:/mnt/ssd0/byeongcheol/Remote_Sensing/oopen_earth_map/sam_vit_h_4b8939.pth 에서 다운로드 받을 수 있다
- 다만 현재 세팅에서는 큰 이미지를 미리 crop 시켜서 skku_tiles_40에 모아둔 뒤, 적용하고 있기 때문에 overlapped stride를 적용할 수 있게끔 변경할 필요가 있다.
- 원본 사진은 vclab@115.145.172.165:/mnt/hdd0/byeongcheol/Remote_Sensing/202403409C02020039.tif 이다.
```python
python test.py --model_name model.pth \
        --model_dir ./model \
        --sam_mask \
        --skku_dir {skku_tile_director}
```



