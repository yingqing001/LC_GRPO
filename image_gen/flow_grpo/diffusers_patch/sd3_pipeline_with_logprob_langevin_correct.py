# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
import math
from typing import Any, Dict, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
# from .sd3_sde_with_logprob import sde_step_with_logprob
# from .flux_pipeline_with_logprob_correct import ode_step, langevin_step_with_logprob
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


@torch.no_grad()
def pipeline_with_logprob_langevin_correct(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    # window_range: tuple[int, int] = None,
    random_end: int = None,
    coeff_type: str = "base_const",
    correction_cfg: bool = True,
    ode_rate: float = 1.0,
    log_mean: bool = True,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ).float()

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []

    # 7. Denoising loop

    if random_end is None:
        random_end = len(timesteps) - 1


    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            latents_dtype = latents.dtype


    

            ## first do a ode step
            latents, sigma_ode, sigma_prev = ode_step(
                self.scheduler, 
                noise_pred.float(), 
                t.unsqueeze(0), 
                latents.float(),
                ode_rate=ode_rate if i < random_end else 1.0,
            )

            if i < random_end:

                next_t = timesteps[i+1]

                ode_end_t = t * (1 - ode_rate) + next_t * ode_rate

                ### assert ode_end_t and sigma_ode is consistent
                sigma_ode_check = ode_end_t / self.scheduler.config.num_train_timesteps
                assert torch.allclose(sigma_ode, sigma_ode_check, rtol=1e-4, atol=1e-4), f"Sigma ODE check failed at step {i}, t={t}, ode_end_t={ode_end_t}, sigma_ode={sigma_ode.mean().item()}, sigma_ode_check={sigma_ode_check.mean().item()}"

                if self.do_classifier_free_guidance and correction_cfg:
                    next_noise_pred = self.transformer(
                        hidden_states=torch.cat([latents] * 2).to(latents_dtype),
                        timestep=torch.cat([ode_end_t.expand(latents.shape[0])] * 2),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    next_noise_pred_uncond, next_noise_pred_text = next_noise_pred.chunk(2)
                    next_noise_pred = (
                        next_noise_pred_uncond
                        + self.guidance_scale
                        * (next_noise_pred_text - next_noise_pred_uncond)
                    )
                else:
                    next_noise_pred = self.transformer(
                        hidden_states=latents.to(latents_dtype),
                        timestep=ode_end_t.expand(latents.shape[0]),
                        encoder_hidden_states=prompt_embeds.chunk(2)[1] if self.do_classifier_free_guidance else prompt_embeds,
                        pooled_projections=pooled_prompt_embeds.chunk(2)[1] if self.do_classifier_free_guidance else pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # print(f"[Debug-Sampling] No correction CFG at step {i}, t={t}, next_t={next_t}")

                latents, log_prob, prev_latents_mean, std_dev_t = langevin_step_general_with_logprob(
                    self=self.scheduler,
                    model_output=next_noise_pred.float(),
                    sigma_ode=sigma_ode,
                    sigma_prev=sigma_prev,
                    sample=latents.float(),
                    noise_level=noise_level,
                    coeff_type=coeff_type,
                    log_mean=log_mean,
                )

         
            
                all_latents.append(latents)
                all_log_probs.append(log_prob)
            # if latents.dtype != latents_dtype:
            #     latents = latents.to(latents_dtype)
            
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    return image, all_latents, all_log_probs





def ode_step(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    ode_rate: float = 1.0,
):
    
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()

    sigma_ode = sigma * (1 - ode_rate) + sigma_prev * ode_rate

    dt = sigma_ode - sigma

    
    # ode step
    prev_sample = sample + model_output * dt

    return prev_sample, sigma_ode, sigma_prev


def langevin_step_general_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    sigma_ode: Union[float, torch.FloatTensor],
    sigma_prev: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    # generator: Optional[torch.Generator] = None,
    noise_level: float = 0.9,
    coeff_type: str = "base_const",
    log_mean: bool = True,
):
    
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()


    
    pred_original_sample = sample - sigma_ode * model_output
    noise_estimate = sample + model_output * (1 - sigma_ode) # predicted x_1 

    std_dev_t = sigma_prev * noise_level

   

    
    if coeff_type == "base_const":

        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * (sigma_prev - std_dev_t ** 2 / (2 * sigma_ode))


    elif coeff_type == "ddim":

        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
        

    corrected_sample_mean = prev_sample_mean
    noise_std = std_dev_t

   

    if prev_sample is None:
        noise = randn_tensor(
                model_output.shape,
                # generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
        corrected_sample = corrected_sample_mean + noise_std * noise
    else:
        corrected_sample = prev_sample

    log_prob = (
        -((corrected_sample.detach() - corrected_sample_mean) ** 2) / (2 * noise_std**2)
        - torch.log(noise_std)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    if log_mean:
        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return corrected_sample, log_prob, corrected_sample_mean, noise_std





       
