# POML

Partially-observed Multi-label Classification \\via Autoencoder-based Matrix Completion

### Requirements

Tensorflow-gpu == 2.3.0, Numpy, sklearn

## CUB

Please download the following CUB-200-2011 data files (https://drive.google.com/drive/folders/1ywd3EKMyNYBdGu2hgOaI_BG8qtU6yUVR?usp=sharing).

Please download the following CUB-200-2011 pretrained CNN models (https://drive.google.com/drive/folders/1DX2SPm6ViCR6OL9PGwLGHavf_4WY6KuO?usp=sharing).

Please download the following CUB-200-2011 pretrained AE models (https://drive.google.com/drive/folders/13G3Rx9r17wr1th9luuuJVg6U0xNgNBTw?usp=sharing).

Training: ratio denotes missing ratio (5:0%, 4:20%, 3:40%, 2:60%, 1: 20%) , hyper_semi: lambda_g, th: alpha (Cub_ft.py: phase 2 in our paper)

(for ratio 3,4,5)
CUDA_VISIBLE_DEVICES=1 python Cub_ft.py --seed ${seed} --ratio 5
CUDA_VISIBLE_DEVICES=1 python Cub_ft.py --seed ${seed} --ratio 2 --max_epoch 20 --hyper_semi 0.5 --th 0.95
CUDA_VISIBLE_DEVICES=1 python Cub_ft.py --seed ${seed} --ratio 1 --max_epoch 70 --hyper_semi 0.4 --th 0.93

If you want to train the scratch model (Cub_init.py: phase 1 in our paper)

CUDA_VISIBLE_DEVICES=1 python Cub_init.py --seed ${seed} --ratio 5


## MS-COCO

Please download the following MSCOCO-caption data files (https://drive.google.com/drive/folders/1pGtoWp4Ut3XIqfpsVTAK0bO34l_LVe_v?usp=sharing).

Please download the following MSCOCO-caption pretrained CNN, AE models (https://drive.google.com/drive/folders/1IcvgSMa5Q2HOlI5jwKaNB1dQx_Q6c1rc?usp=sharing).

CUDA_VISIBLE_DEVICES=1 python COCO_ft.py

If you want to train the scratch model  (Cub_init.py: phase 1 in our paper)

CUDA_VISIBLE_DEVICES=1 python COCO_init.py

##  Open Images

We upload only training code (AE_learning_OpenImages.py) dues to huge amount of data.
In case of Open Images, we implemented based on the code of "https://github.com/hbdat/cvpr20_IMCL/"

You can download OpenImages dataset with  help of "https://github.com/hbdat/cvpr20_IMCL/" (kind manual)

@article{Huynh-mll:CVPR20,
  author = {D.~Huynh and E.~Elhamifar},
  title = {Interactive Multi-Label {CNN} Learning with Partial Labels},
  journal = {{IEEE} Conference on Computer Vision and Pattern Recognition},
  year = {2020}}

Please download the following Open Images pretrained CNN, AE models (https://drive.google.com/drive/folders/1c5HryIlyySDXGXfUjYWWuk6_pcP-XqbT?usp=sharing)
Please download the following Open Images image files into ./TFRecords folder with  help of "https://github.com/hbdat/cvpr20_IMCL/"
Please download the following Open Images data files into ./data/OpenImages (https://drive.google.com/drive/folders/1UkSUPWmel_D49aPLO9PpibaBaslY9etV?usp=sharing)

CUDA_VISIBLE_DEVICES=1 python AE_learning_OpenImages.py
