import logging
import os
import json
import sys
import random
import numpy as np
from utils.execute import PythonExecutor, extract_function_names
from tasks.base import Task



class ListFunction(Task):
    def __init__(
            self, 
            data_path,
            n_train,
            n_test,
            ood_ratio=0.0,
            noise_ratio=0.0,
            mode="base",
            prompt=None
        ):
        super().__init__(data_path, n_train, n_test, ood_ratio, noise_ratio, mode, prompt)
        
    @staticmethod
    def synthetic_data(data_path):
        # get functions
        def get_list_functions():
            function_dict = {}
            with open(os.path.join(data_path, 'list_function.py'), 'r') as f:
                functions = f.readlines()
            line_idx = 0
            while line_idx < len(functions):
                if 'def' in functions[line_idx]:
                    function_item = functions[line_idx]
                    line_idx += 1
                    while 'def' not in functions[line_idx]:
                        function_item += functions[line_idx]
                        line_idx += 1
                        if line_idx >= len(functions):
                            break
                    function_name = extract_function_names(function_item)[0]
                    function_dict[function_name] = function_item
            return function_dict
        
        def generate_input(n, mini=0, maxi=99, min_len=5, max_len=8, type="normal", contains_num=None):
            def normal_input():
                inputs = []
                for _ in range(n):
                    input = [random.randint(mini, maxi) for _ in range(random.randint(min_len, max_len))]
                    inputs.append(input)
                return inputs
            
            def contains(number):
                inputs = []
                for i in range(n):
                    input = [random.randint(mini, maxi) for _ in range(random.randint(min_len, max_len))]
                    for _ in range(random.randint(1, 3)):
                        input.append(number)
                    random.shuffle(input)
                    inputs.append(input)
                return inputs
            
            def duplicate_elements():
                inputs = []
                for _ in range(n):
                    input = [random.randint(mini, maxi) for _ in range(random.randint(min_len, max_len))]
                    for _ in range(random.randint(1, 2)):
                        idx = random.randint(0, len(input)-1)
                        input.append(input[idx])
                    random.shuffle(input)
                    inputs.append(input)
                return inputs

            if type == "normal":
                return normal_input()
            elif type == "contains":
                return contains(contains_num)
            elif type == "duplicate_elements":
                return duplicate_elements()
            else:
                # manually construct inputs
                return []

        python_executor = PythonExecutor()

        def generate_output(function_item, inputs, out_type="normal"):
            outputs = []
            try:
                function_name = extract_function_names(function_item)[0]
                for inp in inputs:
                    output = python_executor.execute(
                        f"{function_item}\n",
                        "x="+str(inp),
                        f"result = {function_name}(x)"
                    )
                    if out_type != "normal":
                        if len(output) == 0:
                            output = [random.randint(0, 20)]
                        else:
                            replace_num = random.randint(1, len(output)//3 + 1)
                            replace_idx = random.sample(range(len(output)), replace_num)
                            for idx in replace_idx:
                                error_out = random.randint(min(output), max(output))
                                while error_out == output[idx]:
                                    error_out = random.randint(min(output), max(output))
                                    if min(output) == max(output):
                                        error_out = max(output) + 1
                                output[idx] = error_out
                    outputs.append(output)
            except Exception as e:
                print(inputs)
                raise e
            return outputs
        
        function_dict = get_list_functions(data_path)
        
        for function_name, function_item in function_dict.items():
            inputs = generate_input(10)
            noise_inputs = generate_input(5)
            test_inputs = generate_input(10)
            outputs = generate_output(function_item, inputs)
            noise_outputs = generate_output(function_item, noise_inputs, "noise")
            test_outputs = generate_output(function_item, test_inputs)
            data_item = {
                "id": function_name,
                "function": function_item,
                "train": {
                    "normal": [{"input": inn, "output": outt} for inn, outt in zip(inputs, outputs)], 
                    "ood": [], 
                    "noise": [{"input": inn, "output": outt} for inn, outt in zip(noise_inputs, noise_outputs)]
                },
                "test": [{"input": inn, "output": outt} for inn, outt in zip(test_inputs, test_outputs)]
            }
            with open(os.path.join(data_path, "list_functions.jsonl"), "a+", encoding="utf-8") as f:
                f.write(json.dumps(data_item) + '\n')
            
