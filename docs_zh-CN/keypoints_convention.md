# 关键点类型

## 总览

MMHuamn3D中的关键点类型整合了各种常用数据集中不同关键点的定义。由于标注数据的过程不尽相同，不同数据集中具有相同名字的关键点可能会对应不同的身体部位。相对的，拥有不同名字的关键点可能对应相同的身体部位。
为了统一不同数据集中不同关键点的对应关系，我们采用`human_data`结构作为转换和存储关键点的基本类型。

## 使用方法

### 不同类型之间的转换

使用 `convert_kps` 函数可以转换不同数据集中的关键点。

为了将`human_data`转换为`coco`的格式，需要指定源类型和目标类型。


```python
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps

keypoints_human_data = np.zeros((100, 190, 3))
keypoints_coco, mask = convert_kps(keypoints_human_data, src='human_data', dst='coco')
assert mask.all()==1
```

如果目标类型是源类型的子集合，输出的`mask`应该全为1。没有对应关系的关键点的置信度会被设置为0, 可以使用`mask`作为关键点的置信度。

### 通过置信度进行转换

如果获得了关键点的置信度，可以使用原始`mask`对其进行标记，然后将信息更新为返回的`mask`。例如，想要将`smpl`的关键点转换为`coco`的关键点，并且已知`left_shoulder`被遮挡。想要在转换过程中继承该信息，可以像如下这样设置原始`mask`并且转换为`coco`类型:

```python
import numpy as np
from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY, convert_kps

keypoints = np.zeros((1, len(KEYPOINTS_FACTORY['smpl']), 3))
confidence = np.ones((len(KEYPOINTS_FACTORY['smpl'])))

# 假设 'left_shoulder' 是无效的.
confidence[KEYPOINTS_FACTORY['smpl'].index('left_shoulder')] = 0

_, conf_coco = convert_kps(
    keypoints=keypoints, confidence=confidence, src='smpl', dst='coco')
_, conf_coco_full = convert_kps(
    keypoints=keypoints, src='smpl', dst='coco')

assert conf_coco[KEYPOINTS_FACTORY['coco'].index('left_shoulder')] == 0
conf_coco[KEYPOINTS_FACTORY['coco'].index('left_shoulder')] = 1
assert (conf_coco == conf_coco_full).all()
```

`mask`表示有效信息，其`dtype`为`uint8`，同时关键点的置信度范围为0到1。
例如，想要将`smpl`的关键点转换为`coco`的关键点, 并且已知`left_shoulder`被遮挡。

```python
confidence = np.ones((len(KEYPOINTS_FACTORY['smpl'])))
confidence[KEYPOINTS_FACTORY['smpl'].index('left_shoulder')] = 0.5
kp_smpl = np.concatenate([kp_smpl, confidence], -1)
kp_smpl_converted, mask = convert_kps(kp_smpl, src='smpl', dst='coco')
new_confidence =  kp_smpl_converted[..., 2:]
assert new_confidence[KEYPOINTS_FACTORY['smpl'].index('left_shoulder')] == 0.5
```

## 支持的种类


以下是支持的关键点种类:
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

`HumanData` 中的前 144 个关键点对应于 `SMPL-X` 中的关键点。后缀为`_extra`的关键点是指从`Jregressor_extra`获得的关键点。 后缀为`_openpose`的关键点是指从`OpenPose`预测中获得的关键点。

`MPI-INF-3DHP`、`Human3.6M` 和 `Posetrack` 中有几个关键点具有相同的名称，但在含义上与 `SMPL-X` 中的关键点不同。 因此，我们添加了一个额外的后缀`head_h36m`来区分这些关键点。

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


### 客制化关键点种类

请参考[customize_keypoints_convention](../docs_zh-CN/customize_keypoints_convention.md).
