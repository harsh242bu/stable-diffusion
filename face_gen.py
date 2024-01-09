import time
from datetime import timedelta
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

from prompt_design import get_prompt_list, get_prompt_list_for_task_string
import params

cache_dir = "/projectnb/ivc-ml/harshk/.cache"

os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface/home")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "huggingface/datasets")
os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, "huggingface/transformers")

model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "stabilityai/stable-diffusion-2-1-base"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_vae_slicing()
# images = pipe([prompt] * 32).images

# pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
# Workaround for not accepting attention shape using VAE for Flash Attention
pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)



def generate_faces(pipe, task):
	# prompt_input, negative_prompt_input = get_prompt_list()
	prompt_input, negative_prompt_input = get_prompt_list(task)
	# prompt_input = [prompt] * num_images
	# negative_prompt_input = [negative_prompt] * num_images


	start_time = time.time()
	# image_list = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=70).images
	image_list = pipe(prompt=prompt_input, negative_prompt=negative_prompt_input, num_inference_steps=60, height=params.image_size, 
					width=params.image_size).images
	
	time_taken = time.time() - start_time
	time_taken = timedelta(seconds=round(time_taken, 2))
	# formatted_time_diff = str(time_taken)
	
	print(f"Time taken: {str(time_taken)}")

	directory = f"face_gen_sd15_data/{task}"
	
	if not os.path.exists(directory):
		os.makedirs(directory)

	num_files = len(os.listdir(directory))

	for i, image in enumerate(image_list):
		counter = i + num_files
		# image.save(f"images/face_{i}.png")
		file_path = f"{directory}/{prompt_input[i]}_{counter}.png"
		
		image.save(file_path)

# For glasses
# for i in range(0, 7):
# 	generate_faces(pipe, params.task_list[i])

# generate_faces(pipe, params.eyes_task[1])

# for l in range(100000):
# 	# generate_faces(pipe, params.eyes_task[0])
# 	generate_faces(pipe, params.eyes_task[0])
# 	generate_faces(pipe, params.eyes_task[1])
# 	print("Loop number done: ", l)

# For sunglasses
# for i in range(7, 9):
# 	generate_faces(pipe, params.task_list[i])


def test_prompts(pos_prompt, neg_prompt):
	prompt_input, negative_prompt_input = get_prompt_list_for_task_string("winking")

	# prompt_input = [pos_prompt] * params.num_images
	# negative_prompt_input = [neg_prompt] * params.num_images

	start_time = time.time()
	# image_list = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=70).images
	image_list = pipe(
		prompt=prompt_input,
		negative_prompt=negative_prompt_input,
		num_inference_steps=60,
		height=params.image_size,
		width=params.image_size,
	).images

	time_taken = time.time() - start_time
	time_taken = timedelta(seconds=round(time_taken, 2))
	# formatted_time_diff = str(time_taken)

	print(f"Time taken: {str(time_taken)}")

	directory = f"face_gen_sd15_winking"

	if not os.path.exists(directory):
		os.makedirs(directory)

	num_images = len(os.listdir(directory))

	for i, image in enumerate(image_list):
		counter = i + num_images
		# image.save(f"images/face_{i}.png")
		file_path = f"{directory}/{prompt_input[i]}_{counter}.png"

		image.save(file_path)

test_prompts(None, None)
