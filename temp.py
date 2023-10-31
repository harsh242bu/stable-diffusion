import os
task = "specs"
prompt_input = "some_input"
file_path = f"glass/{task}/{prompt_input}_5.png"

if not os.path.exists(file_path):
    os.makedirs(os.path.dirname(file_path))
	
