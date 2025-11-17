# IMPUS

Image Morphing with Perceptually-Uniform Sampling Using Diffusion Models

Official code for "IMPUS: Image Morphing with Perceptually-Uniform Sampling Using Diffusion Models" (ICLR 2024).

![teaser](teaser/barbie_oppen.jpg)

## Overview

IMPUS proposes a method for perceptually-uniform sampling along diffusion-based image morphing trajectories. The repository provides reference code and examples to reproduce morph generation and evaluation from the paper.

### Paper
- PDF: https://openreview.net/pdf?id=gG38EBe2S8

## Requirements
- Python 3.9+ (tested with 3.9)
- GPU with at least ~14GB VRAM recommended
- PyTorch 1.13.1 (or compatible)
- Hugging Face Diffusers

Install dependencies:

```powershell
pip install -r requirements.txt
```

The pretrained diffusion weights used in our experiments are from Stable Diffusion v1.4 (CompVis). See model links in the paper and `requirements.txt` for versions.

## Quickstart

- Interactive demo: open and run `IMPUS.ipynb` (recommended for first-time exploration).
- Command-line morphing: use `run_morph.py` to generate morph sequences from two images and a prompt. The script is directly convert from the notebook. 

Example CLI usage:

```powershell
python run_morph.py --dir outputs/morph1 --input_image_1 data/img1.png --input_image_2 data/img2.png --prompt "A portrait of a smiling person"
```


## Files of interest
- `IMPUS.ipynb` — interactive notebook demonstrating training/inference.
- `run_morph.py` — command-line interface for generating morphs.
- `diffuser_helpers_cond_uncond_lora.py` — helper utilities for diffusion models wiht LoRA.
- `metrics_benchmark.py` — evaluation metrics used in experiments.


## Citation
If you use this code or the method in your work, please cite:

```bibtex
@inproceedings{
yang2024impus,
title={{IMPUS}: Image Morphing with Perceptually-Uniform Sampling Using Diffusion Models},
author={Zhaoyuan Yang and Zhengyang Yu and Zhiwei Xu and Jaskirat Singh and Jing Zhang and Dylan Campbell and Peter Tu and Richard Hartley},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=gG38EBe2S8}
}
```

## Acknowledgements

We thank the Hugging Face `diffusers` and the wider community for providing the diffusion model tooling and ecosystem that made this work possible. We also acknowledge the authors and maintainers of the Stable Diffusion checkpoints (CompVis) used in our experiments.
