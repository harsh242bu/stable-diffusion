import time
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

from prompt_design import get_prompt, get_prompt_list
import params

model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "stabilityai/stable-diffusion-2-1-base"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_vae_slicing()
# images = pipe([prompt] * 32).images

pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
# Workaround for not accepting attention shape using VAE for Flash Attention
pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)

# prompt = "a photo of an astronaut riding a horse on mars"
# prompt = "a face of a man with eyes closed"
# prompt, race, gender, task = get_prompt()
# negative_prompt = "animated, cartoon, sketch, painting"
num_images = 130


def generate_faces(pipe, task):
	# prompt_input, negative_prompt_input = get_prompt_list()
	prompt_input, negative_prompt_input = get_prompt_list(task)
	# prompt_input = [prompt] * num_images
	# negative_prompt_input = [negative_prompt] * num_images


	start_time = time.time()
	# image_list = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=70).images
	image_list = pipe(prompt=prompt_input, negative_prompt=negative_prompt_input, num_inference_steps=50, height=params.image_size, 
					width=params.image_size).images
	print(f"Time taken: {time.time() - start_time} seconds")

	directory = f"specs_data/{task}"
	if not os.path.exists(directory):
			os.makedirs(directory)

	for i, image in enumerate(image_list):
		# image.save(f"images/face_{i}.png")
		file_path = f"{directory}/{prompt_input[i]}_{i}.png"
		
		image.save(file_path)

# For glasses
# for i in range(0, 7):
# 	generate_faces(pipe, params.task_list[i])

generate_faces(pipe, params.task_list[0])
# For sunglasses
# for i in range(7, 9):
# 	generate_faces(pipe, params.task_list[i])
