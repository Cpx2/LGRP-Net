# LGRP-Net

Official code repository for **LGRP-Net**, a few-shot image classification method.

> **Paper:** [Add the paper title and publication/preprint link here]  
> **Authors:** [Add author names here]

## Overview

LGRP-Net is evaluated in the few-shot image classification setting. The implementation in this repository includes:

- **Visual feature extraction:** ResNet-12 and Swin-Tiny backbones, with an optional combined ResNet-12/Swin-Tiny backbone.
- **Semantic feature mapping:** the `SemAlign` module maps semantic and visual features into a visual feature space.
- **Prototype refinement:** the `QSWPR` module uses query features to refine support-set class prototypes.
- **Few-shot evaluation:** episodic evaluation on miniImageNet, tieredImageNet, CIFAR-FS, and FC100.

The repository is organized as follows:

```text
LGRP-Net/
├── data/                  # Dataset loading, sampling, and preprocessing code
├── method/
│   ├── train.py           # Training and validation
│   ├── test.py            # Few-shot evaluation
│   ├── QSWPR.py            # Query-weighted prototype refinement
│   └── SemAlign.py         # Semantic-visual feature mapping
├── model/
│   ├── res12.py
│   ├── swin_transformer.py
│   └── Swin_Res12.py       # Combined visual backbone
├── utils/
│   └── utils.py
├── compute_center.py      # Computes class centers
├── requirements.txt
├── .gitignore
└── README.md
```
## Framework

The overall architecture of LGRP-Net is illustrated below.

<p align="center">
  <img src="figures/LGRP-Net.png" width="95%">
</p>

## Datasets

The code supports the following datasets:

- miniImageNet
- tieredImageNet
- CIFAR-FS
- FC100

**Dataset files are not included in this repository.** Please obtain each dataset from its official source and arrange the image files in the directory structure expected by the data-loading code.

The current training and testing scripts contain local dataset-path settings. Before running the code on another machine, update those paths to match your local dataset locations.

## Dependencies

Install the Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The exact Python and package versions used for the reported experiments should be added here after confirming the original experimental environment.

## Pretrained models and semantic resources

Pretrained visual checkpoints and semantic feature files are **not distributed in this repository**.

This project uses resources associated with **SemFew**. Please obtain the corresponding code, pretrained weights, and semantic resources from the original project and follow its access and license terms:

- **SemFew original repository:** [Insert the verified SemFew GitHub URL]

Please retain the original authors' attribution and comply with the licenses governing all third-party code and resources. Citing or linking to an upstream project does not, by itself, grant permission to redistribute its files.

The training and evaluation scripts expect the required checkpoints and semantic files at the paths specified in the code (including the `checkpoints/` and `semantic/` directories). Configure these paths after obtaining the resources.

## Running the code

Run commands from the repository root.

### 1. Compute class centers

```bash
python compute_center.py --dataset MiniImageNet --backbone resnet
```

Supported dataset choices:

```text
MiniImageNet, TieredImageNet, CIFAR-FS, FC100
```

Supported backbone choices:

```text
resnet, swin, swin_res12
```

### 2. Train

For example:

```bash
python -m method.train --dataset MiniImageNet --backbone resnet
```

The training script exposes additional arguments, including `--max-epoch`, `--batch-size`, `--lr`, `--shot`, `--query`, and `--test-batch`. Use:

```bash
python -m method.train --help
```

### 3. Evaluate

For example:

```bash
python -m method.test --dataset MiniImageNet --backbone resnet --shot 1
```

The test script exposes additional arguments, including `--shot`, `--query`, `--test-batch`, and `--backbone`. Use:

```bash
python -m method.test --help
```

**Important:** These commands illustrate the script entry points and argument interface. Before running, make sure the dataset paths, pretrained checkpoint paths, semantic-resource paths, and any required trained model checkpoint paths are configured correctly. The evaluation script also expects the trained fusion module checkpoint produced by training.

## Experimental protocol

The experiments use episodic few-shot classification. The reported evaluation protocol in the paper should be documented here with the final settings, including the number of ways, shots, query examples per class, number of episodes, and confidence-interval calculation.

[Confirm and insert the exact settings used for each reported result.]

## Citation

If you use this code in your research, please cite the LGRP-Net paper:

```bibtex
@article{lgrpnet,
  title   = { [Insert paper title] },
  author  = { [Insert authors] },
  journal = { [Insert venue] },
  year    = { [Insert year] }
}
```

## License

A license for this repository will be added after checking the licenses and redistribution requirements of the third-party components included or adapted in the code. Until then, please do not assume that the repository grants permission to reuse every component.
