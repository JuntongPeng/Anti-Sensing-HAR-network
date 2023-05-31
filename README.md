# Anti-Sensing-HAR-network
> This is the code for course project "Novel anti-sensing communication based on deep learning"

> **you need to prepare .mat data and place them under ```/dataset```**
  
## Tasks Introduction
- Task 1: Efficient Detection Module for HAR
- Task 2: Encoder-Decoder Network with low BER & Encoder hiding sensing feature
- Task 3: Trade-off between privacy and accuracy
  - against fixed sensing module
  - against optimized sensing module

## Requirements
<img src="resources/requirement.png" style="zoom:20%;"  alt="requirement"/>

[download model](https://jbox.sjtu.edu.cn/l/P1dfhZ)

- [x] Task1: sensing module with a classification accuracy of more than 80%
- [x] Task2: encoder-decoder network with a BER of less than 10%
- [x] Task3: anti-sensing encoder 
  - [x] against fixed sensing module
  - [x] against optimized sensing module

## Usage
- Task 1

  Training
  ```
  python task1.py --model_dir <model dir>
  ```
  Inference
  ```
  python task1_inference.py --model_dir <model dir>
  ```

- Task 2

  Training
  ```
  python task2.py --model_dir <model dir> --bit_len <bit length> --lr <learning rate> --random_mask <True or False>
  ```
  Inference
  ```
  python task2_inference.py --model_dir <model dir>
  ```

- Task 3

  Training
  ```
  python task3.py --sensing_module_dir <sensing module dir> --encdec_dir <encdec dir> --stage <enc_dec or sensing_module> --bit_len <bit length> --random_mask <True or False> --mask_ratio <number between 0 and 1> --preamble <True or False>
  ```
