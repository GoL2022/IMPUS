# IMPUS: Image Morphing with Perceptually-Uniform Sampling Using Diffusion Models  
     
> Official repository of *IMPUS: Image Morphing with Perceptually-Uniform Sampling Using Diffusion Models*, ICLR 2024.

> **This repository is under reconstruction...**

![teaser](teaser/barbie_oppen.jpg)

### [Paper](https://openreview.net/pdf?id=gG38EBe2S8)

## TODO 

- [ ] Release Benchmark & pretrained checkpoints  

## Environment setup 

This code was tested with Python 3.9, [Pytorch](https://pytorch.org/) 1.13.1. based on [huggingface / diffusers](https://github.com/huggingface/diffusers#readme). The pretrained diffusion model is from [Stable Diffusion v-1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4). Install package using [requirements.txt](requirements.txt) by ```pip install -r requirements.txt```.


The code requires at least 14GB VRAM.

## Quickstart

Both training and inference for IMPUS are available at [IMPUS.ipynb](IMPUS.ipynb). Images in the notebook is for simple demo (replace with high resolution image generate better results), we will update with more examples later. 
## Reference
to do: add code reference
## Citation 
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
