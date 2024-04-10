from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.loaders import AttnProcsLayers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler #, DDIMInverseScheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
def extract_lora_diffusers(unet, device, dtype=None, rank=4):
    ### ref: https://github.com/huggingface/diffusers/blob/4f14b363297cf8deac3e88a3bf31f59880ac8a96/examples/dreambooth/train_dreambooth_lora.py#L833
    ### begin lora
    # Set correct lora layers, default rank is 4
    unet_lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = LoRAAttnProcessor

        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
        ).to(device)
    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # self.unet.requires_grad_(True)
    unet.requires_grad_(False)
    for param in unet_lora_layers.parameters():
        param.requires_grad_(True)
    # self.params_to_optimize = unet_lora_layers.parameters()
    ### end lora
    if not dtype:
        unet = unet.to(dtype)
        unet_lora_layers = unet_lora_layers.to(dtype)
    return unet, unet_lora_layers
def predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, guidance_scale=7.5, cross_attention_kwargs={}, scheduler=None, lora_v=False, half_inference=False, mask_uncon = False):
    batch_size = noisy_latents.shape[0]
    # for both conditioned & unconditioned generation
    
    if guidance_scale == 1.:
        latent_model_input = noisy_latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        #noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings,
        #                  cross_attention_kwargs=cross_attention_kwargs).sample
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings[batch_size:],   cross_attention_kwargs=cross_attention_kwargs).sample
        noise_pred_text = noise_pred#.chunk(2)
        return noise_pred_text
    else:
        latent_model_input = torch.cat([noisy_latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        if mask_uncon:
            noise_pred_text.register_hook(lambda grad: grad * torch.zeros_like(grad).float())
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    noise_pred = noise_pred.float()
    return noise_pred



@torch.no_grad()
def sample_model(unet, scheduler, c, scale, start_code):
    """Sample the model"""
    prev_noisy_sample = start_code
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = unet(torch.cat([prev_noisy_sample] * 2), t, encoder_hidden_states=c).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)

            prev_noisy_sample = scheduler.step(noise_pred, t, prev_noisy_sample).prev_sample
    return prev_noisy_sample


from diffusers.pipeline_utils import DiffusionPipeline
from typing import Callable, List, Optional, Union
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from functools import partial

def backward_ddim(x_t, alpha_t: "alpha_t", alpha_tm1: "alpha_{t-1}", eps_xt, sigma_t=0):
    """ from noise to image"""
    #print('sigma_t', sigma_t)
    eps_t = torch.randn_like(x_t)

    x_hat_0 =  (x_t - (1 - alpha_t)**0.5 * eps_xt) / (alpha_t**0.5)

    direct_to_x_t = (1 - alpha_tm1 - sigma_t ** 2.0) ** 0.5 * eps_xt

    random_noise = sigma_t * eps_t

    x_tm1 = (alpha_tm1 ** 0.5) * x_hat_0 + direct_to_x_t + random_noise
    return x_hat_0, x_tm1

def forward_ddim(x_t, alpha_t: "alpha_t", alpha_tp1: "alpha_{t+1}", eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)

class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler_sampling: DDIMScheduler,
            scheduler_inversion: DDIMScheduler,
            safety_checker: StableDiffusionSafetyChecker = None,
            feature_extractor: CLIPFeatureExtractor = None,
            unet_uncond: UNet2DConditionModel = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler_sampling=scheduler_sampling,
            scheduler_inversion=scheduler_inversion,
            # safety_checker=safety_checker,
            # feature_extractor=feature_extractor,
        )
        self.unet_uncond = unet_uncond
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    @torch.inference_mode()
    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance
    @torch.inference_mode()
    def backward_diffusion(
            self,
            use_old_emb_i=25,
            text_embeddings=None,
            old_text_embeddings=None,
            new_text_embeddings=None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            reverse_process: True = False,
            return_x_hat_0 = False,
            eta = 0,
            quality_boosting_steps = 0,
            use_source_uncon=False,
            use_unet_uncon=False,
            **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        if reverse_process:
            self.scheduler = self.scheduler_inversion
        else:
            self.scheduler = self.scheduler_sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False
        if quality_boosting_steps or reverse_process:
            do_quality_boost = False
        else:
            do_quality_boost = True
        x_hat_0_lst = []

        for i, t in enumerate(
                self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance and (not use_source_uncon and not use_unet_uncon) else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # ddim
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
            )
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t

            # predict the noise residual
            if (not use_source_uncon and not use_unet_uncon):
                # predict the noise residual using 
                noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                if use_source_uncon:
                    noise_pred_uncond = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0), cross_attention_kwargs = {'scale':0}
                    ).sample

                    noise_pred_text = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0), cross_attention_kwargs = {'scale':1}
                    ).sample       
                elif use_unet_uncon:
                    noise_pred_uncond = self.unet_uncond(
                    latent_model_input, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0), cross_attention_kwargs = {'scale':1}
                    ).sample

                    noise_pred_text = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0), cross_attention_kwargs = {'scale':1}
                    ).sample 
                
                else:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )


            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)


            #print('len(timesteps_tensor)', len(timesteps_tensor))
            #print('timesteps_tensor', timesteps_tensor)
            if quality_boosting_steps != 0 and len(timesteps_tensor) - i == quality_boosting_steps:
                do_quality_boost = True

            sigma_t = eta * (((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5) * ((1 - alpha_prod_t / alpha_prod_t_prev) ** 0.5) if do_quality_boost else 0
            #print('alpha_prod_t_prev', alpha_prod_t_prev)
            #print('alpha_prod_t', alpha_prod_t)
            x_hat_0, latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
                sigma_t = sigma_t
            )
            if return_x_hat_0:
                x_hat_0_lst.append(x_hat_0)
        #if likelihood_computation:
        #    return likelihood_lst, latents
        if return_x_hat_0:
            return x_hat_0_lst, latents
        else:
            return latents


    def backward_diffusion_train(
            self,
            use_old_emb_i=25,
            text_embeddings=None,
            old_text_embeddings=None,
            new_text_embeddings=None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            reverse_process: True = False,
            **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        if reverse_process:
            self.scheduler = self.scheduler_inversion
        else:
            self.scheduler = self.scheduler_sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False

        for i, t in enumerate(
                self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
            )
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

            # ddim
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        return latents

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs) -> List["PIL_IMAGE"]:
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i: i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image) -> List["PIL_IMAGE"]:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image
