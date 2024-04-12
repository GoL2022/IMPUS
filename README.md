# IMPUS: Image Morphing with Perceptually-Uniform Sampling Using Diffusion Models  
     
> Official repository of *IMPUS: Image Morphing with Perceptually-Uniform Sampling Using Diffusion Models*, ICLR 2024.

> **This repository is under reconstruction...**

![teaser](teaser/barbie_oppen.jpg)

### [Paper](https://openreview.net/pdf?id=gG38EBe2S8)

## TODO 

- [ ] Release Benchmark & pretrained checkpoints  

## Environment setup 

This code was tested with Python 3.9, [Pytorch](https://pytorch.org/) 1.13.1. based on [huggingface / diffusers](https://github.com/huggingface/diffusers#readme). The pretrained diffusion model is from [Stable Diffusion v-1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4). Install package using [requirements.txt](requirements.txt) by ```pip install -r requirements.txt```.

```bash
conda env create -f environment.yml
conda activate IMPUS
```

The code requires at least 14GB VRAM.

## Quickstart

Both training and inference for IMPUS are available at [IMPUS.ipynb](IMPUS.ipynb). Images in the notebook is for simple demo (a little blur, replace with high resolution image generate better results), we will update with more examples later. 
## Reference
to do: add code reference
## Citation 
```bibtex
@article{yang2024impus,
  title={IMPUS: Image Morphing with Perceptually-Uniform Sampling Using Diffusion Models},
  author={Yang, Zhaoyuan, Zhengyang Yu, Zhiwei Xu, Jaskirat Singh, Jing Zhang, Dylan Campbell, Peter Tu, and Richard Hartley},
  journal={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=gG38EBe2S8}
}
```
