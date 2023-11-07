import numpy as np

import params

# prompt = f"Face of a {race} {gender} with {task}"


age_list = []



def get_prompt():
    race = params.race_list[0]
    gender = params.gender_list[2]
    task = params.task_list[0]
    prompt = f"Face of a {race} {gender} with {task}"
    return prompt, race, gender, task

def get_prompt_list(task, num_images=params.num_images):
    prompt_list = []

    for _ in range(num_images):
        race = np.random.choice(params.race_list)
        gender = np.random.choice(params.gender_list)
        prompt = f"Face of a {race} {gender} with {task}"
        if task == params.eyes_task[1]:
            # prompt = f"A portrait of a {race} {gender}"
            prompt = f"Face of a {race} {gender}"
        prompt_list.append(prompt)
    
    prompt_list.sort()
    neg_prompt_list = [params.negative_prompt]*num_images

    return prompt_list, neg_prompt_list


# def get_prompt_list():
#     race = race_list[0]
#     gender = gender_list[2]
#     prompt_list = []
#     for task in task_list[2:]:
#         temp = [f"Face of a {race} {gender} with {task}"]*10
#         prompt_list = prompt_list + temp
#     neg_prompt_list = [negative_prompt]*len(prompt_list)
#     return prompt_list, neg_prompt_list