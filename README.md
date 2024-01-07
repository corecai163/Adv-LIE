# Adversarially Regularized Low-Light Image Enhancement

## dataset

### LOL datasets
Please download the LOL-v1 and LOL-v2 from
https://drive.google.com/drive/folders/1Kev7Np9hWEYHDxlcZXa7vIjBrX6O-Yry?usp=sharing

## Project Setup

pip install -r requirements.txt

## Usage

### Train

```
sh train.sh
```

### Test

We use PSNR and SSIM as the metrics for evaluation. Evaluate the model on the corresponding dataset using the test config.

For the evaluation, 
use the following command lines:
```
python test_LOLv1_v2_real.py
```
```
python test_LOLv2_synthetic.py 
```

## Citation Information

If you find the project useful, please cite:

```
@inproceedings{advLIE,
  title={Adversarially Regularized Low-Light Image Enhancement},
  author={William Y. Wang, Lisa Liu, and Pingping Cai},
  booktitle={International Conference on Multimedia Modeling (MMM)},
  year={2024}
}
```


## Acknowledgments
This source code is inspired by [SNR-aware Low-Light Image Enhancement](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance), [MIRNet](https://github.com/swz30/MIRNet).

