<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.png">
    <img src="assets/logo_light.png" alt="Logo of LmPT" width="500">
  </picture>
</p>

<p align="center">
  <a href="https://pierreoo.github.io/projects/lmpt/"><img src="https://img.shields.io/badge/Project%20Page-LmPT-blue.svg?style=for-the-badge" alt="Project Page" height="24"></a>
  <a href="https://arxiv.org/abs/2602.02808"><img src="https://img.shields.io/badge/arXiv-2602.02808-b31b1b.svg?style=for-the-badge" alt="arXiv" height="24"></a>
  <a href="https://drive.google.com/drive/folders/163P-exuQuSQRwdnW7-RXOxAWZKaDhkZi?usp=drive_link"><img src="https://img.shields.io/badge/Resources-Google%20Drive-seagreen.svg?style=for-the-badge" alt="Datasets and Pre-trained Model" height="24"></a>
  <a href="https://biomedicalimaging.org/2026/"><img src="https://img.shields.io/badge/Conference-ISBI%20Oral%202026-rebeccapurple.svg?style=for-the-badge" alt="ISBI Oral 2026" height="24"></a>
</p>

**LmPT (LandmarkPointTransformer)** is a transformer-based framework built upon the [Pointcept](https://github.com/Pointcept/Pointcept) codebase, extending it with engines for anatomical landmark detection.  

This repository includes the methods as introduced in **LmPT: Conditional Point Transformer for Anatomical Landmark Detection on 3D Point Clouds**.


## <br>Datasets

Available for download via [KeyboneNetCross](https://drive.google.com/drive/folders/1Vggo3RTWlzEmCd50pi6vApICekawpDeA?usp=drive_link).  
Place the content under the [`data/KeyboneNetCross`](data/KeyboneNetCross) directory.

### Dog Femur Dataset

This dataset, introduced in this work, includes 14 models of dog femurs from 7 dogs of different breeds and sizes (7 left, 7 right) under `FBD`.  
With `mesh` and `pcds` representations, each model includes 11 anatomical landmark `annotations`.


### Human Femur Dataset

This dataset includes 20 models of human femurs from different subjects (10 left, 10 right) under `FBH`.  
With `mesh` and `pcds` representations, each model includes 22 anatomical landmark `annotations`.

The representations are derived from the [VSDFullBodyBoneModels](https://github.com/RWTHmediTEC/VSDFullBodyBoneModels) dataset by RWTHmediTEC.


## <br>Pre-trained model

A pre-trained, cross-species LmPT-v2 model is available for download via [LmPT-v2](https://drive.google.com/drive/folders/1IuwTHCS3cPyEMHf0-bHac9gJeQrvvKE8?usp=drive_link).  
Place the content under the [`exp/keybonenetcross/lfv2_cross`](exp/keybonenetcross/lfv2_cross) directory.


## <br>Quick Start

### Training

**Train from scratch** using a configuration file from `configs`, which will create an experiment folder in `exp` with training outputs.
```bash
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
```

For example:
```bash
sh scripts/train.sh -p python -g 1 -d keybonenetcross -c lfv2 -n scratch
```


### Testing

**Test a model** using the experiment name and corresponding config from a trained checkpoint.
```bash
sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
```

For example, to test the pre-trained LmPT-v2 model:
```bash
sh scripts/test.sh -p python -g 1 -d keybonenetcross -n lfv2_cross -w model_best
```


## <br>Licenses

<a href="https://github.com/Pierreoo/LandmarkPointTransformer/blob/main/LICENSE"><img src="https://img.shields.io/badge/LmPT%20Code%20License-BSD--2--Clause-slategray.svg?style=for-the-badge" alt="LmPT Code License: BSD-2-Clause" height="22"></a>
<a href="https://github.com/Pierreoo/LandmarkPointTransformer/blob/main/LICENSE_pointcept"><img src="https://img.shields.io/badge/Pointcept%20Code%20License-MIT-slategray.svg?style=for-the-badge" alt="Pointcept Code License: MIT" height="22"></a>  
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/LmPT%20Dataset%20License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=for-the-badge" alt="LmPT Dataset License: CC BY-NC-SA 4.0" height="22"></a>
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/VSDFullBodyBoneModels%20Dataset%20License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=for-the-badge" alt="VSDFullBodyBoneModels Dataset License: CC BY-NC-SA 4.0" height="22"></a>

This repository incorporates a modified copy of Pointcept, whose original code remains under the MIT License. Modifications made for LmPT are licensed under the BSD 2-Clause License.


## <br>Citation

If you find *LmPT* useful to your research, please consider citing:
```bibtex
@inproceedings{bastico2026lmpt,
  author    = {Bastico, Matteo and Onghena, Pierre and Ryckelynck, David and Marcotegui, Beatriz and Velasco-Forero, Santiago and Cort{\'e}, Laurent and Robine-Decourcelle, Caroline and Decenci{\`e}re, Etienne},
  booktitle = {2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  title     = {{LmPT}: Conditional Point Transformer for Anatomical Landmark Detection on {3D} Point Clouds},
  year      = {2026},
  pages     = {1--5},
  publisher = {IEEE},
  doi       = {10.1109/ISBI61048.2026.11515617}
}
```
