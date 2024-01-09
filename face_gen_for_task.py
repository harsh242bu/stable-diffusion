import time
import argparse
from datetime import timedelta
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

from prompt_design import get_prompt_list
import params
import constant

cache_dir = "/projectnb/ivc-ml/harshk/.cache"

os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface/home")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "huggingface/datasets")
os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, "huggingface/transformers")


def generate_faces_for_face_task(args, pipe, task):
    prompt_input, negative_prompt_input = get_prompt_list(task)


    start_time = time.time()
    image_list = pipe(prompt=prompt_input, negative_prompt=negative_prompt_input, num_inference_steps=60, height=params.image_size, 
                    width=params.image_size).images

    time_taken = time.time() - start_time
    time_taken = timedelta(seconds=round(time_taken, 2))
    # formatted_time_diff = str(time_taken)

    print(f"Time taken: {str(time_taken)}")

    return image_list, prompt_input

def write_image_to_directory(args, image_list, task, prompt_input):
    directory = args.output_dir
    if task is not None:
        task = task.replace(" ", "_").lower()
        directory = f"{args.output_dir}/{task}"
	
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_files = len(os.listdir(directory))

    for i, image in enumerate(image_list):
        counter = i + num_files
        fname = f"{prompt_input[i]}_{counter}.png"
        fname = fname.replace(" ", "_").lower()
        file_path = f"{directory}/{fname}"
        
        image.save(file_path)

# For glasses
# for i in range(0, 7):
# 	generate_faces(pipe, params.task_list[i])

# For sunglasses
# for i in range(7, 9):
# 	generate_faces(pipe, params.task_list[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Face Generation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=constant.MODE_EYE_TASK,
        help="Mode for face generation",
        choices=[constant.MODE_EYE_TASK, constant.MODE_TEST_PROMPT],
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model ID for Stable Diffusion",
		choices=["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1", "stabilityai/stable-diffusion-2-1-base"],
    )
    parser.add_argument(
		"--iters",
		type=int,
		default=99999,
		help="Number of iterations for face generation",
    )
    parser.add_argument(
		"--pos_prompt",
		type=str,
		default=None,
		help="Positive prompt for face generation",
    )
    parser.add_argument(
		"--neg_prompt",
		type=str,
        default=None,
		help="Negative prompt for face generation",
    )
    parser.add_argument(
		"--num_images",
		type=int,
        default=None,
		help="Number of images to generate",
    )
	
    args = parser.parse_args()
    print("Arguments: ", args)

    if args.output_dir is None:
        print("Error: Output directory not specified. Exiting...")
        exit(1)

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()

    # pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    # Workaround for not accepting attention shape using VAE for Flash Attention
    pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)

    if args.mode == constant.MODE_EYE_TASK:
        
        for iter in range(args.iters):
            image_list, prompt_input = generate_faces_for_face_task(args, pipe, params.eyes_task[0])
            write_image_to_directory(args, image_list, params.eyes_task[0], prompt_input)

            image_list, prompt_input = generate_faces_for_face_task(args, pipe, params.eyes_task[1])
            write_image_to_directory(args, image_list, params.eyes_task[1], prompt_input)
            print("Loop number done: ", iter)

    elif args.mode == constant.MODE_TEST_PROMPT:
        if args.pos_prompt is None or args.neg_prompt is None:
            args.pos_prompt = "full face"
            args.neg_prompt = "animated, cartoon, sketch, painting"

        if args.num_images is not None:
            params.num_images = args.num_images

        prompt_input = [args.pos_prompt] * params.num_images
        negative_prompt_input = [args.neg_prompt] * params.num_images

        start_time = time.time()
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

        write_image_to_directory(args, image_list, None, prompt_input)
