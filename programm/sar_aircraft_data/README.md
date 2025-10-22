SAR Aircraft Detection Dataset (SADD)
====================================

This folder is a placeholder and downloader instructions for the SAR aircraft detection dataset (SADD) published by HUST RSLab.

Source repository: `https://github.com/hust-rslab/SAR-aircraft-data`

Label format: YOLO txt per image — `cls cx cy w h` with normalized coordinates, as described in the source README.

How to prepare locally
----------------------
1) Create the expected structure:

```
src/datasets/sar_aircraft_data/
  images/          # put all SADD images here (e.g., .jpg/.png)
  labels/          # put all YOLO .txt labels here
```

2) Download the dataset archive from the source and extract images and labels:

- Repository page: `https://github.com/hust-rslab/SAR-aircraft-data`
- BaiduYun link and extraction code (from the repo):

```
https://pan.baidu.com/s/11SBfMkGszu3Lr_Pe1lIEkg
Extraction Code: d2uw
```

Please see the original repo README for details and citation requirements.

Notes
-----
- This dataset will be used to train the MiniObjNet gate and for calibration of INT8.
- File names of images and labels must match (e.g., `img1.jpg` and `img1.txt`).


