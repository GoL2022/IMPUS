# This is based on https://github.com/toshas/torch-fidelity/
 
import numpy as np
import torch
from tqdm import tqdm

#from torch_fidelity.generative_model_base import GenerativeModelBase
from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.utils import sample_random, batch_interp, create_sample_similarity, \
    prepare_input_descriptor_from_input_id, prepare_input_from_descriptor
device = 'cuda:0'
KEY_METRIC_PPL_RAW = 'perceptual_path_length_raw'
KEY_METRIC_PPL_MEAN = 'perceptual_path_length_mean'
KEY_METRIC_PPL_STD = 'perceptual_path_length_std'

class morpher:
    def _init_(self, pipe, img1, img2, emb_1, emb_2, max_scale=3, min_scale=1.5, use_unet_uncon=True, sampling_steps=16, reverse_steps = 250):
        self.pipe = pipe 
        
        self.image_1 = Image.open(img1).convert("RGB")
        init_image_1 = self.load_img(self.image_1).to(device).unsqueeze(0)
        self.init_latent_1 = pipe.vae.config.scaling_factor * pipe.vae.encode(init_image_1).latent_dist.sample()
        self.image_2 = Image.open(img2).convert("RGB")
        init_image_2 = self.load_img(self.image_2).to(device).unsqueeze(0)
        self.init_latent_2 = pipe.vae.config.scaling_factor * vae.encode(init_image_2).latent_dist.sample()
        
        self.emb_1 = emb_1
        self.emb_2 = emb_2
        self.start_code1 =  pipe.forward_diffusion(
            latents=self.init_latent_1,
            text_embeddings=emb_1[1].unsqueeze(0),
            guidance_scale=1,
            num_inference_steps=reverse_steps,
        )
        self.start_code2 =  pipe.forward_diffusion(
            latents=self.init_latent_2,
            text_embeddings=emb_2[1].unsqueeze(0),
            guidance_scale=1,
            num_inference_steps=reverse_steps,
        )
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.sample_steps = sampling_steps
        self.use_unet_uncon = use_unet_uncon
        self.reverse_steps = reverse_steps
    def latents_to_imgs(self, latents):
        x = self.pipe.decode_image(latents)
        x = self.pipe.torch_to_numpy(x)
        x = self.pipe.numpy_to_pil(x)
        return x
    def load_img(self, image, target_size=512):
        """Load an image, resize and output -1..1"""
    
        tform = transforms.Compose([
            #transforms.Resize((512,512)),
            #transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
        image = tform(image)
        return 2. * image - 1.
    #Spherical linear interpolation
    def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
        '''
        Spherical linear interpolation
        Args:
            t (float/np.ndarray): Float value between 0.0 and 1.0
            v0 (np.ndarray): Starting vector
            v1 (np.ndarray): Final vector
            DOT_THRESHOLD (float): Threshold for considering the two vectors as
                                   colineal. Not recommended to alter this.
        Returns:
            v2 (np.ndarray): Interpolation vector between v0 and v1
        '''
        from torch import lerp
        c = False
        if not isinstance(v0,np.ndarray):
            c = True
            v0 = v0.detach().cpu().numpy()
        if not isinstance(v1,np.ndarray):
            c = True
            v1 = v1.detach().cpu().numpy()
        # Copy the vectors to reuse them later
        v0_copy = np.copy(v0)
        v1_copy = np.copy(v1)
        # Normalize the vectors to get the directions and angles
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)
        # Dot product with the normalized vectors (can't use np.dot in W)
        dot = np.sum(v0 * v1)
        # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
        if np.abs(dot) > DOT_THRESHOLD:
            return lerp(t, v0_copy, v1_copy)
        # Calculate initial angle between v0 and v1
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        # Angle at timestep t
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0 
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0_copy + s1 * v1_copy
        if c:
            res = torch.from_numpy(v2).to(device)
        else:
            res = v2
        return res
    def quick_sample(self, emb, scale, start_code):
        # return a PIL image object
        x0 = self.pipe.backward_diffusion(
                    latents=start_code,
                    text_embeddings=emb,
                    guidance_scale=scale,
                    num_inference_steps=self.sampling_steps,
                    use_unet_uncon=self.use_unet_uncon)
        return self.pipe.decode_image(x0)
    def morph(self, alpha):
        new_emb = alpha * self.emb_2 + (1 - alpha) * self.emb_1
        scale = self.max_scale - np.abs(alpha - 0.5) * (self.max_scale-self.min_scale) * 2.0
        start_code = self.slerp(alpha, self.start_code1, self.start_code2)
        return self.quick_sample(new_emb, scale, start_code)

def lpips(img1, img2):
    import lpips
    loss_fn_vgg = lpips.LPIPS(net='alex').to(device)
    return loss_fn_vgg(img1, img2)

def calculate_ppl(morpher, similarity, discard_percentile_lower=1, num_samples=50, num_classes=0, discard_percentile_higher=99,epsilon=1e-4, verbose=False):
    """
    Inspired by https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
    """


    vassert(num_samples >= 0, 'Model can be unconditional (0 classes) or conditional (positive)')
    vassert(type(epsilon) is float and epsilon > 0, 'Epsilon must be a small positive floating point number')
    vassert(type(num_samples) is int and num_samples > 0, 'Number of samples must be positive')

    vassert(discard_percentile_lower is None or 0 < discard_percentile_lower < 100, 'Invalid percentile')
    vassert(discard_percentile_higher is None or 0 < discard_percentile_higher < 100, 'Invalid percentile')
    if discard_percentile_lower is not None and discard_percentile_higher is not None:
        vassert(0 < discard_percentile_lower < discard_percentile_higher < 100, 'Invalid percentiles')

    alphas = np.random.rand(num_samples)

    distances = []

    with tqdm(disable=not verbose, leave=False, unit='samples', total=num_samples,
            desc='Perceptual Path Length') as t, torch.no_grad():
        for idx in range(0, num_samples, 1):
            alpha_im1 = alphas[idx]
            alpha_i = alphas[idx] + epsilon
            x_im1 = morpher.morph(alpha_im1)
            x_i = morpher.morph(alpha_i)
            sim = similarity(x_im1, x_i)
            dist_lat_e01 = sim / (epsilon ** 2)
            distances.append(dist_lat_e01.cpu().numpy())

    distances = np.concatenate(distances, axis=0)

    cond, lo, hi = None, None, None
    if discard_percentile_lower is not None:
        lo = np.percentile(distances, discard_percentile_lower, interpolation='lower')
        cond = lo <= distances
    if discard_percentile_higher is not None:
        hi = np.percentile(distances, discard_percentile_higher, interpolation='higher')
        cond = np.logical_and(cond, distances <= hi)
    if cond is not None:
        distances = np.extract(cond, distances)

    out = {
        KEY_METRIC_PPL_MEAN: float(np.mean(distances)),
        KEY_METRIC_PPL_STD: float(np.std(distances)),
        KEY_METRIC_PPL_RAW: float(distances)
    }

    vprint(verbose, f'Perceptual Path Length: {out[KEY_METRIC_PPL_MEAN]} ± {out[KEY_METRIC_PPL_STD]}')

    return out