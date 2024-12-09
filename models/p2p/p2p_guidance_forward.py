import torch

from models.p2p.attention_control import register_attention_control
from utils.utils import init_latent,latent2image, image2latent
import os
from PIL import Image



import torch.nn as nn
from torch.fft import fftn, ifftn, fftshift, ifftshift
from abc import ABC, abstractmethod
import numpy as np
class FourierTransformModel(nn.Module):
        def __init__(self,device):#, mask_type, device):
            super(FourierTransformModel, self).__init__()
            # self.mask_type = mask_type
            self.device = device

        def get_mask(self, mags, cutoff):
            # (mag_r, mag_g, mag_b) = mags
            # h, w = mag_r.shape
            # C =len(mags)
            N, C, h, w = mags.shape 
            y, x = torch.meshgrid(torch.arange(-h//2, h//2), torch.arange(-w//2, w//2), indexing='ij')
            distance = torch.sqrt(x**2 + y**2)
            mask = (distance < cutoff).float()
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(1, C, 1, 1).to(self.device)
            
            return mask

        def forward(self, x, cutoff):
            channel_fft = fftn(x, dim=(-2, -1),norm='ortho')
            channel_fft_shifted = fftshift(channel_fft, dim=(-2, -1))
        
            _, c, h, w = x.shape
            
            mask = self.get_mask(x, cutoff)
            # import pdb; pdb.set_trace()
            channel_fft_shifted_filtered = channel_fft_shifted * mask
            
            # Shift back the zero frequency component
            channel_fft_filtered = ifftshift(channel_fft_shifted_filtered, dim=(-2, -1))
            
            # channel_fft_log = torch.log(channel_fft_filtered)            
            return channel_fft_filtered
    



class ConditioningMethod(ABC):
    def __init__(self):#, operator, noiser, **kwargs):
        # self.operator = operator
        # self.noiser = noiser
        pass

    
    def grad_and_value(self, x_prev, x_pred, image_gt, t,cutoff_schedule=None,model=None,cond_model=None,  **kwargs):
        cutoff = cutoff_schedule[t.cpu()]

        # image_tensor = image2tensor(model, image_gt)

        #reshape images to 256x256

        difference = cond_model(image_gt, cutoff) \
                - cond_model(x_pred, cutoff)
        ###check image_gt requires_grad
        # with torch.autograd.set_detect_anomaly(True):
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad, norm

class ModelPosteriorSampling(ConditioningMethod):
    def __init__(self, model, cond_model, schedule, start_bound, end_bound,scale_schedule,scale_start, scale_end ,**kwargs):
        super().__init__()

        self.scale_schedule = scale_schedule  #kwargs.get('scale_schedule', None)
        if self.scale_schedule=="fixed":
            self.scale = scale_start
        else:
            # Having learning rate for each time step
            # Parameters
            a = scale_start    # Initial value
            b = scale_end    # Minimum value
            T = 1000   # Total time steps
            time_steps = np.linspace(0, T, T)
            annealing_values = 0.5 * (a + b) + 0.5 * (a - b) * np.cos(np.pi * time_steps / T)
            self.scale = torch.tensor(annealing_values).to("cuda")
        
        self.model = model
        self.cond_model = cond_model
        if schedule == 'linear':
            self.cutoff_schedule = torch.linspace(start_bound, end_bound, 1000).flip(0)
        elif schedule == 'cosine':
            self.cutoff_schedule = (end_bound - (end_bound - start_bound) * torch.exp(-torch.linspace(0, 5, 1000))).flip(0)
        else:
            raise NotImplementedError("Only linear schedule is currently supported for ModelPosteriorSampling.")


    def conditioning(self, target_latent_prev, x_pred, image_gt, target_latent, t, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=target_latent_prev, x_pred=x_pred, image_gt=image_gt, \
                                t=t, cutoff_schedule=self.cutoff_schedule, model=self.model,cond_model=self.cond_model , **kwargs)
        
        if self.scale_schedule=="fixed":
            target_latent = target_latent - (norm_grad * (self.scale))
            scaled_norm = norm_grad * (self.scale)
        else:
            target_latent = target_latent - (norm_grad * (self.scale[t]))
            scaled_norm = norm_grad * (self.scale[t])
        
        # print("distance at time t {} is {}".format(t, norm))
        return target_latent, scaled_norm



def pred_x_start(model, model_output, timestep: int, sample):
        alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        return pred_original_sample

def latent2image_withgrad(model, latents):
    latents = 1 / 0.18215 * latents
    image = model.decode(latents)['sample']
    return image

def image2tensor(model, image):
    if type(image) is Image:
        image = np.array(image)
    if type(image) is torch.Tensor and image.dim() == 4:
        latents = image
    else:
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(model.device)
    return image

def p2p_guidance_diffusion_step_fgps(model, controller, latents, context, t, guidance_scale, low_resource=False,save_path=None,branch_name=None,image_gt=None):

    prev_latent = latents

    

    low_resource = False
    if low_resource:
        noise_pred_uncond = model.unet(prev_latent, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(prev_latent, t, encoder_hidden_states=context[1])["sample"]
    else:
        with torch.no_grad():
            latents_input = torch.cat([prev_latent.detach()] * 2)
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    if branch_name is not None:
        prev_latent = prev_latent.requires_grad_(True)
        noise_pred = noise_pred.requires_grad_(False)
    
    latents = model.scheduler.step(noise_pred, t, prev_latent)["prev_sample"]
    
    x_start_latent = pred_x_start(model,noise_pred, t, prev_latent)   
    # x_start_pred_img = latent2image_withgrad(model=model.vae, latents=x_start_latent)
    gt_latent = image2latent(model.vae, image_gt)
    gt_latent = torch.cat([gt_latent]*latents.shape[0])  #TODO:changed here for full prompts
    
   

    # if branch_name is not None:
    #     #same the image
    #     #extract the image name from save_path
    #     with torch.no_grad():
    #         x_start_pred_img_pil = Image.fromarray(latent2image(model.vae, x_start_latent)[0])

    #     name = os.path.basename(save_path)
    #     #remove the extension
    #     name = os.path.splitext(name)[0]
    #     path = os.path.dirname(save_path) + f"/{name}/{branch_name}/x_start_pred_{t}.png"
    #     if not os.path.exists(os.path.dirname(path)):
    #         os.makedirs(os.path.dirname(path))
    #     x_start_pred_img_pil.save(path)

    
    if branch_name is not None:

        fourier_model = FourierTransformModel(model.device)
        fourier_model = fourier_model.to(model.device)
        fourier_model.eval()
        cond_model = fourier_model
        scale_schedule = "cosine"
        
        if branch_name == "source":
            freq_start = 10
            freq_end = 100
            scale_start = 3.5
            scale_end = 0.05
        else:
            freq_start = 10
            freq_end = 50
            scale_start = 1.0
            scale_end = 0.01
        # if t>900:
        #     print(f"scale schedule is {scale_schedule}, scale start is {scale_start}, scale end is {scale_end}, freq start is {freq_start}, freq end is {freq_end}")
        if branch_name == "source":
            conditioning_method = ModelPosteriorSampling(model, cond_model, 'cosine', freq_start, freq_end, scale_schedule=scale_schedule, scale_start=scale_start, scale_end=scale_end)
            latents, norm_grad = conditioning_method.conditioning(target_latent_prev=prev_latent, x_pred=x_start_latent, image_gt=gt_latent, target_latent=latents, t=t)   #TODO:changed here for full prompts
        # elif branch_name == "target" and t>400:
        #     conditioning_method = ModelPosteriorSampling(model, cond_model, 'cosine', freq_start, freq_end, scale_schedule=scale_schedule, scale_start=scale_start, scale_end=scale_end)
        #     latents, norm = conditioning_method.conditioning(target_latent_prev=prev_latent, x_pred=x_start_latent, image_gt=gt_latent, target_latent=latents, t=t)
    ###add fgps code here if branch_name is source

    # latents_grad = torch.cat((latents_grad[:1],latents[1:]))
    latents = controller.step_callback(latents)
    return latents, norm_grad

def p2p_guidance_diffusion_step_fgps_target(model, controller, latents, context, t, guidance_scale, low_resource=False,save_path=None,branch_name=None,image_gt=None,source_latents=None):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    # latents = torch.concat((latents[:1] - (source_latents * scale[t]),latents[1:]))
    latents = torch.concat((latents[:1] - source_latents,latents[1:]))
    latents = controller.step_callback(latents)
    return latents

def p2p_guidance_forward_fgps(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None,
    save_path=None,
    branch_name=None,
    image_gt=None
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    # max_length = text_input.input_ids.shape[-1]
    # if uncond_embeddings is None:
    #     uncond_input = model.tokenizer(
    #         [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    #     )
    #     uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    # else:
    #     uncond_embeddings_ = None

    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    latents_list = []
    source_latents = []
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    latents_list.append(latents)
    for i, t in enumerate(model.scheduler.timesteps):
        # if uncond_embeddings_ is None:
        #     context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        # else:
        #     context = torch.cat([uncond_embeddings_, text_embeddings])
        
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents, norm_grad = p2p_guidance_diffusion_step_fgps(model, controller, latents, context, t, guidance_scale, low_resource=False,save_path=save_path,branch_name=branch_name,image_gt=image_gt)
        source_latents.append(norm_grad[:1])
        latents_list.append(latents)
    
    latents = latents.detach()
    return latents, latent, source_latents, latents_list
@torch.no_grad()
def p2p_guidance_forward_fgps_target(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None,
    save_path=None,
    branch_name=None,
    image_gt=None,
    source_latents=None
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = p2p_guidance_diffusion_step_fgps_target(model, controller, latents, context, t, guidance_scale, low_resource=False,save_path=save_path,branch_name=branch_name,image_gt=image_gt,source_latents=source_latents[i])
    
    latents = latents.detach()
    return latents, latent



###########################################################################




def p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False,save_path=None,branch_name=None):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]


    latents = controller.step_callback(latents)
    return latents

@torch.no_grad()
def p2p_guidance_forward(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None,
    save_path=None,
    branch_name=None,
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latents_list = []
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    latents_list.append(latents)
    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False,save_path=save_path,branch_name=branch_name)
        latents_list.append(latents)
    
    return latents, latent, latents_list


@torch.no_grad()
def p2p_guidance_forward_single_branch(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat([torch.cat([uncond_embeddings[i],uncond_embeddings_[1:]]), text_embeddings])
        latents = p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    return latents, latent


def direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]))
    
    latents = controller.step_callback(latents)
    return latents


def direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    if add_offset:
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]+noise_loss[1:]))
    latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def direct_inversion_p2p_guidance_forward(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    noise_loss_list = None,
    add_offset=True
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = direct_inversion_p2p_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, noise_loss_list[i],low_resource=False,add_offset=add_offset)
        
    return latents, latent

@torch.no_grad()
def direct_inversion_p2p_guidance_forward_add_target(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps: int = 50,
    guidance_scale = 7.5,
    generator = None,
    noise_loss_list = None,
    add_offset=True
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        
        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = direct_inversion_p2p_guidance_diffusion_step_add_target(model, controller, latents, context, t, guidance_scale, noise_loss_list[i],low_resource=False,add_offset=add_offset)
        
    return latents, latent
