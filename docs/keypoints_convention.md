# Keypoints convention

## Overview

Our convention tries to consolidate the different keypoints definition across various
commonly used datasets. Due to differences in data-labelling procedures, keypoints across datasets with the same name might not map to semantically similar locations on the human body. Conversely, keypoints with different names might correspond to the same location on the human body. To unify the different keypoints correspondences across datasets, we adopted the `human_data` convention
as the base convention for converting and storing our keypoints.

## How to use

### Converting between conventions

Keypoints can be converted between different conventions easily using the `convert_kps` function.

To convert a `human_data` keypoints to `coco` convention, specify the source and
destination convention for conversion.

```python
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps

keypoints_human_data = np.zeros((100, 190, 3))
keypoints_coco, mask = convert_kps(keypoints_human_data, src='human_data', dst='coco')
assert mask.all()==1
```

The output `mask` should be all ones if the `dst` convention is the subset of the `src` convention.
You can use the `mask` as the confidence of the keypoints since those keypoints with no correspondence are set to a default value with 0 confidence.


### Converting with confidence

If you have confidential information of your keypoints, you can use an original mask to mark it, then the information will be updated into the returned mask.
E.g., you want to convert a `smpl` keypoints to `coco` keypoints, and you know its `left_shoulder` is occluded. You want to carry forward this information during the converting. So you can set an original_mask and convert it to `coco` by doing:

```python
import numpy as np
from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY, convert_kps

keypoints = np.zeros((1, len(KEYPOINTS_FACTORY['smpl']), 3))
confidence = np.ones((len(KEYPOINTS_FACTORY['smpl'])))

# assume that 'left_shoulder' point is invalid.
confidence[KEYPOINTS_FACTORY['smpl'].index('left_shoulder')] = 0

_, conf_coco = convert_kps(
    keypoints=keypoints, confidence=confidence, src='smpl', dst='coco')
_, conf_coco_full = convert_kps(
    keypoints=keypoints, src='smpl', dst='coco')

assert conf_coco[KEYPOINTS_FACTORY['coco'].index('left_shoulder')] == 0
conf_coco[KEYPOINTS_FACTORY['coco'].index('left_shoulder')] = 1
assert (conf_coco == conf_coco_full).all()
```

Our mask represents valid information, its dtype is uint8, while keypoint confidence usually ranges from 0 to 1.
E.g., you want to convert a `smpl` keypoints to `coco` keypoints, and you know its `left_shoulder` is occluded. You want to carry forward this information during the converting. So you can set an original_mask and convert it to `coco` by doing:

```python
confidence = np.ones((len(KEYPOINTS_FACTORY['smpl'])))
confidence[KEYPOINTS_FACTORY['smpl'].index('left_shoulder')] = 0.5
kp_smpl = np.concatenate([kp_smpl, confidence], -1)
kp_smpl_converted, mask = convert_kps(kp_smpl, src='smpl', dst='coco')
new_confidence =  kp_smpl_converted[..., 2:]
assert new_confidence[KEYPOINTS_FACTORY['smpl'].index('left_shoulder')] == 0.5
```

## Supported Conventions


These are the supported conventions:
  - [AGORA](#agora)
  - [COCO](#coco)
  - [COCO-WHOLEBODY](#coco-wholebody)
  - [CrowdPose](#crowdpose)
  - [GTA-Human](#gta-human)
  - [Human3.6M](#human36m)
  - human_data
  - [HybrIK](#hybrik)
  - [LSP](#lsp)
  - [MPI-INF-3DHP](#mpi-inf-3dhp)
  - [MPII](#mpii)
  - openpose
  - [PennAction](#pennaction)
  - [PoseTrack18](#posetrack18)
  - [PW3D](#pw3d)
  - [SMPL](#smpl)
  - [SMPL-X](#smplx)


### HUMANDATA

The first 144 keypoints in HumanData correspond to that in `SMPL-X`.
Keypoints with suffix `_extra` refer to those obtained from Jregressor_extra.
Keypoints with suffix `_openpose` refer to those obtained from `OpenPose` predictions.

There are several keypoints from `MPI-INF-3DHP`, `Human3.6M` and `Posetrack` that has the same name but were semantically different from keypoints in `SMPL-X`. As such, we added an extra suffix to differentiate those keypoints i.e. `head_h36m`.

### AGORA

<details>
<summary align="right"><a href="https://arxiv.org/pdf/2104.14643.pdf">AGORA (CVPR'2021)</a></summary>

```bibtex
@inproceedings{Patel:CVPR:2021,
  title = {{AGORA}: Avatars in Geography Optimized for Regression Analysis},
  author = {Patel, Priyanka and Huang, Chun-Hao P. and Tesch, Joachim and Hoffmann, David T. and Tripathi, Shashank and Black, Michael J.},
  booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition ({CVPR})},
  month = jun,
  year = {2021},
  month_numeric = {6}
}
```

</details>

### COCO

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48">COCO (ECCV'2014)</a></summary>

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

</details>


### COCO-WHOLEBODY

<details>
<summary align="right"><a href="https://arxiv.org/abs/2007.11858.pdf">COCO-Wholebody (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>


### CrowdPose

<details>
<summary align="right"><a href="https://arxiv.org/pdf/1812.00324.pdf">CrowdPose (CVPR'2019)</a></summary>

```bibtex
@article{li2018crowdpose,
  title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
  author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
  journal={Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition ({CVPR})},
  year={2019}
}
```

</details>

### Human3.6M


<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6682899/">Human3.6M (TPAMI'2014)</a></summary>

```bibtex
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {IEEE Computer Society},
  volume = {36},
  number = {7},
  pages = {1325-1339},
  month = {jul},
  year = {2014}
}
```

</details>


### GTA-Human

<details>
<summary align="right"><a href="https://gta-human.com/">GTA-Human (arXiv'2021)</a></summary>

```bibtex
@article{cai2021playing,
  title={Playing for 3D Human Recovery},
  author={Cai, Zhongang and Zhang, Mingyuan and Ren, Jiawei and Wei, Chen and Ren, Daxuan and Li, Jiatong and Lin, Zhengyu and Zhao, Haiyu and Yi, Shuai and Yang, Lei and others},
  journal={arXiv preprint arXiv:2110.07588},
  year={2021}
}
```

</details>


### HybrIK

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/CVPR2021/html/Li_HybrIK_A_Hybrid_Analytical-Neural_Inverse_Kinematics_Solution_for_3D_Human_CVPR_2021_paper.html">HybrIK (CVPR'2021)</a></summary>

```bibtex
@inproceedings{li2020hybrikg,
  author = {Li, Jiefeng and Xu, Chao and Chen, Zhicun and Bian, Siyuan and Yang, Lixin and Lu, Cewu},
  title = {HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation},
  booktitle={CVPR 2021},
  pages={3383--3393},
  year={2021},
  organization={IEEE}
}
```

</details>

### LSP


<details>
<summary align="right"><a href="http://sam.johnson.io/research/publications/johnson10bmvc.pdf">LSP (BMVC'2010)</a></summary>

```bibtex
@inproceedings{johnson2010clustered,
  title={Clustered Pose and Nonlinear Appearance Models for Human Pose Estimation.},
  author={Johnson, Sam and Everingham, Mark},
  booktitle={bmvc},
  volume={2},
  number={4},
  pages={5},
  year={2010},
  organization={Citeseer}
}
```
</details>

### MPI-INF-3DHP

<details>
<summary align="right"><a href="https://arxiv.org/pdf/1611.09813.pdf">MPI_INF_3DHP (3DV'2017)</a></summary>

```bibtex
@inproceedings{mono-3dhp2017,
 author = {Mehta, Dushyant and Rhodin, Helge and Casas, Dan and Fua, Pascal and Sotnychenko, Oleksandr and Xu, Weipeng and Theobalt, Christian},
 title = {Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision},
 booktitle = {3D Vision (3DV), 2017 Fifth International Conference on},
 url = {http://gvv.mpi-inf.mpg.de/3dhp_dataset},
 year = {2017},
 organization={IEEE},
 doi={10.1109/3dv.2017.00064},
}
```

</details>


### MPII


<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

### PoseTrack18


<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Andriluka_PoseTrack_A_Benchmark_CVPR_2018_paper.html">PoseTrack18 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{andriluka2018posetrack,
  title={Posetrack: A benchmark for human pose estimation and tracking},
  author={Andriluka, Mykhaylo and Iqbal, Umar and Insafutdinov, Eldar and Pishchulin, Leonid and Milan, Anton and Gall, Juergen and Schiele, Bernt},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5167--5176},
  year={2018}
}
```

</details>

### OpenPose


<details>
<summary align="right"><a href="https://arxiv.org/pdf/1812.08008v2.pdf">OpenPose(TPMAI'2019)</a></summary>

```bibtex
@article{8765346,
  author = {Z. {Cao} and G. {Hidalgo Martinez} and T. {Simon} and S. {Wei} and Y. A. {Sheikh}},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title = {OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2019}
}
```

</details>

### PennAction


<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_iccv_2013/papers/Zhang_From_Actemes_to_2013_ICCV_paper.pdf
">PennAction(ICCV'2013)</a></summary>

```bibtex
@inproceedings{zhang2013,
  title={From Actemes to Action: A Strongly-supervised Representation for Detailed Action Understanding},
  author={Zhang, Weiyu and Zhu, Menglong and Derpanis, Konstantinos},
  booktitle={Proceedings of the International Conference on Computer Vision},
  year={2013}
}
```

</details>


### SMPL


<details>
<summary align="right"><a href="https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf">SMPL(ACM'2015)</a></summary>

```bibtex
@article{SMPL:2015,
      author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
      title = {{SMPL}: A Skinned Multi-Person Linear Model},
      journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
      month = oct,
      number = {6},
      pages = {248:1--248:16},
      publisher = {ACM},
      volume = {34},
      year = {2015}
    }
```

</details>


### SMPL-X


<details>
<summary align="right"><a href="https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf
">SMPL-X(CVPR'2019)</a></summary>

```bibtex
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: {3D} Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10975--10985},
  year = {2019}
}
```

</details>


### Customizing keypoint convention

Please refer to [customize_keypoints_convention](./customize_keypoints_convention.md).
