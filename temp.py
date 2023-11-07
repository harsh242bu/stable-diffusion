import os
import params
from prompt_design import get_prompt_list

task = "specs"
prompt_input = "some_input"
file_path = f"glass/{task}/{prompt_input}_5.png"

# if not os.path.exists(file_path):
#     os.makedirs(os.path.dirname(file_path))

# dir_path = f"face_gen_data/{params.task_list[0]}"
# print("list: ", len(os.listdir(dir_path)))

# import torch
# print(torch.cuda.is_available())

plist, _ = get_prompt_list(params.task_list[0])
plist.sort()
print(plist)