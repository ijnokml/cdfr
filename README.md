# FRCD

This is for [Feature Refine Change Detection (FRCD) ](https://ieeexplore.ieee.org/document/10210603)in remote sensing images.

Dataset shodld be in ```./dataset/CCD_real```

How to train: 
```bash
python -m torch.distributed.launch --nproc_per_node 2 onlytrain.py
```

evaluating:
```bash
python onlyeval.py
```
The drawing script will be updated soon. 

We will upload the trained model ```.pth``` in the future. 

The configs can be found in ./configs and 

2 is the number of GPUs you want to use. Every GPU have 8 batch size.

The codes are based on [SwinTransformer official implementation](https://github.com/microsoft/Swin-Transformer) and MMLab's mmsegmentation and MMCV. 

Environment that may used to run the code is as follows:
```bash
# Name                    Version                   Build  Channel
cudatoolkit               11.3.1               h2bc3f7f_2    defaults
einops                    0.4.1                    pypi_0    pypi
mmcv-full                 1.4.8                    pypi_0    pypi
mmsegmentation            0.23.0                   pypi_0    pypi
numpy                     1.21.5           py39he7a7128_1    defaults
opencv-python             4.5.5.64                 pypi_0    pypi
pillow                    9.0.1            py39h22f2fdc_0    defaults
python                    3.9.12               h12debd9_1    defaults
pytorch                   1.11.0          py3.9_cuda11.3_cudnn8.2.0_0    pytorch
scikit-image              0.19.2           py39h51133e4_0    defaults
scikit-learn              1.2.2                    pypi_0    pypi
scipy                     1.7.3            py39hc147768_0    defaults
tensorboard               2.8.0                    pypi_0    pypi
timm                      0.6.12                   pypi_0    pypi
tqdm                      4.64.0                   pypi_0    pypi
yacs                      0.1.8                    pypi_0    pypi
```
