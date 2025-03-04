# DDAMFN
Paper: A Dual-Direction Attention Mixed Feature Network for Facial Expression Recognition
https://www.mdpi.com/2079-9292/12/17/3595

Data processing: [https://qq742971636.blog.csdn.net/article/details/136841591](https://qq742971636.blog.csdn.net/article/details/141253148)(Thank the author of this blog for his interpretation and contribution to the paper. For further learning needs, you can subscribe to his column.)

Our new version DDAMFN++ is available in the folder https://github.com/simon20010923/DDAMFN/tree/main/DDAMFN%2B%2B, which includes new alignment code refer to https://github.com/biubug6/Pytorch_Retinaface and optimized by SAM ([Sharpness aware minimization for efficiently improving generalization](https://arxiv.org/abs/2010.01412)).
The trained RAF-DB and AffectNet model is available in DDAMFN++/checkpoints_ver2.0, which is not our best.

下载 `AffectNet.zip`，重命名：

```text
.
├── train
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   ├── 5
│   ├── 6
│   ├── 7
│   └── 8
└── val
    ├── 1
    ├── 2
    ├── 3
    ├── 4
    ├── 5
    ├── 6
    ├── 7
    └── 8
```

训练：

```bash
cd DDAMFN++
python affectnet_train_sam_opt_v2.0.py --aff_path F:\Datasets\AffectNet --num_class 8 --batch_size 32
```
