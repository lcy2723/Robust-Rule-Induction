import random
from tqdm import tqdm
import json
import os
from utils.response import InductionResponse
from utils.execute import PythonExecutor, extract_function_names, extract_function
import concurrent.futures

class Task:
    def __init__(
        self, 
        data_path, 
        n_train, 
        n_test, 
        ood_ratio=0.0, 
        noise_ratio=0.0,
        mode="base",
        prompt=None,
        sample_ratio=1.0,
        **kwargs
    ):
        self.data_path = data_path
        with open(self.data_path, "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]
        self.data = self.data[:int(len(self.data) * sample_ratio)]
        self.eval_length = len(self.data)
        self.n_train = n_train
        self.n_test = n_test
        self.ood_ratio = ood_ratio
        self.noise_ratio = noise_ratio
        self.mode = mode
        if self.mode == "base":
            if self.noise_ratio > 0:
                self.prompt = prompt["noisy"]
            else:
                self.prompt = prompt["base"]
        elif self.mode == "cot" or self.mode=="sc":
            self.prompt = prompt["cot"]
        else:
            self.prompt = prompt
        self.refine_round = 3
    
    def get_train_examples(self, data, position=None):
        # 0: normal, 1: ood, 2: noise
        normal = data["train"]["normal"][:int(self.n_train * (1 - self.ood_ratio-self.noise_ratio))]
        ood = data["train"]["ood"][:int(self.n_train * self.ood_ratio)]
        noise = data["train"]["noise"][:int(self.n_train * self.noise_ratio)]
        examples = normal + ood + noise
        labels = [0] * len(normal) + [1] * len(ood) + [2] * len(noise)
        if position is not None:
            if position == -1:
                noise_positions = random.sample(range(self.n_train), len(noise))
            else:
                noise_positions = [position]
            examples = normal + ood
            labels = [0] * len(normal) + [1] * len(ood)
            for i, noise_position in enumerate(noise_positions):
                examples.insert(noise_position, noise[i])
                labels.insert(noise_position, 2)
        assert len(examples) == self.n_train
        return (examples, labels)

    def get_test_examples(self, data):
        return data["test"]["normal"][:self.n_test], [0] * self.n_test

    # rewirte in the subclass
    def check_correctness(self, res, output):
        return res == output
    
    def get_input(self, item):
        return str(item["input"])

    def get_response(self, config_path, output_path, model, position=None):
        with open(config_path, "r") as f:
            config = json.load(f)
        reasoner = InductionResponse(config, model)
        for item in tqdm(self.data, desc="Generating responses"):
            # print(prompt)
            examples, labels = self.get_train_examples(item, position)
            if self.mode == "sc":
                examples = "\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples])
                prompt = self.prompt.format(examples=examples)
                item["examples"] = examples
                item["labels"] = labels
                item["response"] = []
                for i in range(5):
                    response = reasoner.generate_response(prompt, temperature=1.0)
                    item["response"].append(response)
            elif self.mode == "sr":
                item["response"] = []
                item["rule_candidates"] = []
                item["feedback"] = []
                item["examples"] = examples
                item["labels"] = labels
                for i in range(self.refine_round):
                    # generate response
                    if i == 0:
                        prompt = self.prompt["sr_base"].format(examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples]))
                        response = reasoner.generate_response(prompt)
                        item["response"].append(response)
                    else:
                        feedback = "\n".join(item["feedback"])
                        prompt = self.prompt["sr_iterative"].format(rule=item["rule_candidates"][-1], examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples]), feedback=feedback)
                        response = reasoner.generate_response(prompt)
                        item["response"].append(response)
                    if i == self.refine_round - 1:
                        item["selected_response"] = item["response"][-1]
                        break
                    # generate feedback
                    try:
                        rule = extract_function(response)
                    except Exception as e:
                        rule = "# no valid rule generated"
                    item["rule_candidates"].append(rule)
                    prompt = self.prompt["sr_feedback"].format(rule=rule, examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples]))
                    response = reasoner.generate_response(prompt, system_message="You are a helpful assistant.")
                    item["feedback"].append(response)
                    if "NO FEEDBACK" in response:
                        item["selected_response"] = item["response"][-1]
                        break
            else:
                examples = "\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples])
                prompt = self.prompt.format(examples=examples)
                response = reasoner.generate_response(prompt)
                item["examples"] = examples
                item["labels"] = labels
                item["response"] = response
            with open(output_path, "a+", encoding='utf-8') as f:
                f.write(json.dumps(item) + "\n")

    def get_response_parallel(self, config_path, output_path, model, position=None):
        with open(config_path, "r") as f:
            config = json.load(f)
        def process_item(item):
            reasoner = InductionResponse(config, model)
            examples, labels = self.get_train_examples(item, position)
            if self.mode == "sc":
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    prompt = self.prompt.format(examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples]))
                    futures = [executor.submit(reasoner.generate_response, 
                        prompt, temperature=1.0) for _ in range(5)]
                    responses = [f.result() for f in concurrent.futures.as_completed(futures)]
                item.update({
                    "examples": examples,
                    "labels": labels,
                    "response": responses
                })
            elif self.mode == "sr":
                item["response"] = []
                item["rule_candidates"] = []
                item["feedback"] = []
                item["examples"] = examples
                item["labels"] = labels
                for i in range(self.refine_round):
                    # generate response
                    if i == 0:
                        prompt = self.prompt["sr_base"].format(examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples]))
                        response = reasoner.generate_response(prompt)
                        item["response"].append(response)
                    else:
                        feedback = "\n".join(item["feedback"])
                        prompt = self.prompt["sr_iterative"].format(rule=item["rule_candidates"][-1], examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples]), feedback=feedback)
                        response = reasoner.generate_response(prompt)
                        item["response"].append(response)
                    if i == self.refine_round - 1:
                        item["selected_response"] = item["response"][-1]
                        return item
                    # generate feedback
                    try:
                        rule = extract_function(response)
                    except Exception as e:
                        rule = "# no valid rule generated"
                    item["rule_candidates"].append(rule)
                    prompt = self.prompt["sr_feedback"].format(rule=rule, examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples]))
                    response = reasoner.generate_response(prompt, system_message="You are a helpful assistant.")
                    item["feedback"].append(response)
                    if "NO FEEDBACK" in response:
                        item["selected_response"] = item["response"][-1]
                        return item
            else:
                prompt = self.prompt.format(examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples]))
                response = reasoner.generate_response(prompt)
                item["examples"] = examples
                item["labels"] = labels
                item["response"] = response

            return item

        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
            processed_items = list(tqdm(
                executor.map(process_item, self.data),
                total=len(self.data),
                desc="Generating responses in parallel"
            ))

        with open(output_path, "a+", encoding='utf-8') as f:
            for item in processed_items:
                f.write(json.dumps(item) + "\n")
                            
    def convert_to_batch(self, output_path, model, position=None, batch_count=3):
        batched_data = []
        print("position", position)
        for batch_index in range(batch_count):
            for index, item in enumerate(self.data):
                examples, labels = self.get_train_examples(item, position)
                examples = "\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples])
                prompt = self.prompt.format(examples=examples)
                item["examples"] = examples
                item["labels"] = labels
                batched_data.append({
                    "custom_id": os.path.basename(output_path).replace(".jsonl", "") + f"_{index}_batch_{batch_index}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert Python programmer."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.7 if self.mode == "sc" else 0.0,
                        "max_tokens": 2048
                    }
                })
        from datetime import datetime
        with open(os.path.dirname(output_path) + f"/{model}_{datetime.now().strftime('%m_%d')}.jsonl", "a+") as f:
            for item in batched_data:
                f.write(json.dumps(item) + "\n")

        
    def evaluate(self, data_path):
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        python_executor = PythonExecutor()
        error_count = 0
        for item in tqdm(data, desc="Evaluating"):
            # for sc
            if self.mode == "sc":
                for test_example in item["test"]:
                    test_example["reses"] = []
                for resp in item["response"]:
                    try:
                        function_string = extract_function(resp)
                        function_name = extract_function_names(function_string)[0]
                        for test_example in item["test"]:
                            input_data = self.get_input(test_example)
                            test_example["reses"].append(python_executor.execute(
                                f"{function_string}\n",
                                "x=" + self.get_input(test_example),
                                f"result = {function_name}(x)"
                            ))
                    except Exception as e:
                        error_count += 1
                        for test_example in item["test"]:
                            test_example["reses"].append("error_occurred")
                # now every test example has 5 results, conduct voting
                for test_example in item["test"]:
                    res_count = [0] * 5
                    for i in range(5):
                        for res in test_example["reses"]:
                            if test_example["reses"][i] == res:
                                res_count[i] += 1
                    test_example["res"] = test_example["reses"][res_count.index(max(res_count))]
            # for base, cot
            else:
                if self.mode == "srr" or self.mode == "sr":
                    response_key = "selected_response"
                else:
                    response_key = "response"
                try:
                    function_string = extract_function(item[response_key])
                    function_name = extract_function_names(function_string)[0]
                    for test_example in item["test"]:
                        input_data = self.get_input(test_example)
                        test_example["res"] = python_executor.execute(
                            f"{function_string}\n",
                            "x=" + input_data,
                            f"result = {function_name}(x)"
                        )
                except Exception as e:
                    error_count += 1
                    for test_example in item["test"]:
                        test_example["res"] = "error_occurred"
        item_correct = 0
        item_count = 0
        task_correct = 0
        task_count = 0
        item_accuracies = []
        task_accuracies = []
        task_complete = []
        for (i, item) in enumerate(data):
            all_correct = True
            for test_example in item["test"]:
                item_count += 1
                if self.check_correctness(test_example["res"], test_example["output"]):
                    item_correct += 1
                else:
                    all_correct = False
            if all_correct:
                task_correct += 1
                task_complete.append(1)
            else:
                task_complete.append(0)

            task_count += 1
            if (i + 1) % self.eval_length == 0:
                item_accuracy = item_correct / item_count
                task_accuracy = task_correct / task_count
                item_accuracies.append(item_accuracy)
                task_accuracies.append(task_accuracy)
                item_correct = 0
                item_count = 0
                task_correct = 0
                task_count = 0
        print("error count: ", error_count)
        # add mean and std to the end
        mean_item_accuracy = sum(item_accuracies) / len(item_accuracies)
        std_item_accuracy = sum([(x - mean_item_accuracy) ** 2 for x in item_accuracies]) / len(item_accuracies)
        std_item_accuracy = std_item_accuracy ** 0.5
        mean_task_accuracy = sum(task_accuracies) / len(task_accuracies)
        std_task_accuracy = sum([(x - mean_task_accuracy) ** 2 for x in task_accuracies]) / len(task_accuracies)
        std_task_accuracy = std_task_accuracy ** 0.5
        item_accuracies.append(mean_item_accuracy)
        item_accuracies.append(std_item_accuracy)
        task_accuracies.append(mean_task_accuracy)
        task_accuracies.append(std_task_accuracy)
        print(item_accuracies, task_accuracies)
        if not os.path.exists(os.path.dirname(data_path) + "/consistency_score"):
            os.makedirs(os.path.dirname(data_path) + "/consistency_score")
        with open(os.path.dirname(data_path) + "/consistency_score/completed.jsonl", "a+") as f:
            f.write(json.dumps({
                "file_name": os.path.basename(data_path),
                "completed": task_complete
            }) + "\n")
        return item_accuracies, task_accuracies


    def run_srr(self, config_path, output_path, model, position=None):
        with open(config_path, "r") as f:
            config = json.load(f)
        python_executor = PythonExecutor()
        workers_setteing = {0: 5, 1: 20, 2: 20}
        # add reload function
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                processed_items = [json.loads(line) for line in f.readlines()]
            # find the last round
            last_round = max([item["response"][-1]["round"] for item in processed_items]) + 1
        else:
            last_round = 0
        def process_item(item, round_index):
            if "selected_response" in item:
                return item
            reasoner = InductionResponse(config, model)
            examples, labels = self.get_train_examples(item, position)
            if round_index == 0: # initial round
                item["response_candidates"] = []
                item["response"] = []
                item["examples"] = examples
                item["labels"] = labels
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    samples1, samples2 = examples[:5], examples[5:]
                    prompt1 = self.prompt["cot"].format(examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in samples1]))
                    prompt2 = self.prompt["cot"].format(examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in samples2]))
                    prompt = self.prompt["cot"].format(examples="\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples]))
                    future1 = executor.submit(
                        reasoner.generate_response, 
                        prompt1,
                        temperature=0.7)
                    future2 = executor.submit(
                        reasoner.generate_response, 
                        prompt2,
                        temperature=0.7)
                    future3 = executor.submit(
                        reasoner.generate_response, 
                        prompt,
                        temperature=0.7)
                    response1, response2, response = future1.result(), future2.result(), future3.result()
                item["response"].extend([
                    {"round": round_index, "response": response1, "examples": samples1, "correct": [], "wrong": [], "prompt": prompt1},
                    {"round": round_index, "response": response2, "examples": samples2, "correct": [], "wrong": [], "prompt": prompt2},
                    {"round": round_index, "response": response, "examples": examples, "correct": [], "wrong": [], "prompt": prompt}
                ])
            else:
                last_response = item["response_candidates"][-1]["rule"]
                last_right = item["response_candidates"][-1]["correct"]
                last_wrong = item["response_candidates"][-1]["wrong"]
                right_examples = random.sample(last_right, min(5, len(last_right)))
                wrong_examples = random.sample(last_wrong, min(3, len(last_wrong)))
                if len(right_examples) != 0:
                    right_examples = "\n".join([f"Input: {ex['input']}    Output: {ex['output']}" for ex in right_examples])
                else:
                    right_examples = "No right examples."
                wrong_examples = "\n".join([f"Input: {ex['input']}    Expected output: {ex['output']}    Your output: {ex['res']}" for ex in wrong_examples])
                prompt = self.prompt["iterative"].format(rule=last_response, right_examples=right_examples, wrong_examples=wrong_examples)
                response = reasoner.generate_response(prompt, temperature=0.7)
                item["response"].append({"round": round_index, "response": response, "correct": [], "wrong": [], "prompt": prompt})
            
            return item
        
        for round_index in range(last_round, self.refine_round):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers_setteing[round_index]) as executor:
                if round_index == 0:
                    processed_items = list(tqdm(
                        executor.map(lambda item: process_item(item, round_index), self.data),
                        total=len(self.data),
                        desc=f"Generating responses in round {round_index}"
                    ))
                else:
                    processed_items = list(tqdm(
                        executor.map(lambda item: process_item(item, round_index), processed_items),
                        total=len(processed_items),
                        desc=f"Generating responses in round {round_index}"
                    ))
                    
            with open(output_path, "w", encoding='utf-8') as f:
                for item in processed_items:
                    f.write(json.dumps(item) + "\n")
            # evaluate the responses for this round
            python_executor = PythonExecutor()
            for item in tqdm(processed_items, desc=f"Evaluating responses for round {round_index}"):
                if "selected_response" in item:
                    continue
                for resp in item["response"]:
                    if resp["round"] == round_index:
                        resp["execution"] = []
                        try:
                            function_string = extract_function(resp["response"])
                            function_name = extract_function_names(function_string)[0]
                            for test_example in item["examples"]:
                                input_data = self.get_input(test_example)
                                exe_res = python_executor.execute(
                                    f"{function_string}\n",
                                    "x=" + input_data,
                                    f"result = {function_name}(x)"
                                )
                                resp["execution"].append({"input": input_data, "res": exe_res, "output": test_example["output"]})
                            resp["rule"] = function_string
                        except Exception as e:
                            resp["rule"] = "# no valid rule generated"
                            for test_example in item["examples"]:
                                resp["execution"].append({"input": "", "res": "no output", "output": test_example["output"]})
                    for exe_res in resp["execution"]:
                        if self.check_correctness(exe_res["res"], exe_res["output"]):
                            resp["correct"].append(exe_res)
                        else:
                            resp["wrong"].append(exe_res)
                    resp["accuracy"] = len(resp["correct"]) / len(item["examples"])
                # select the best response in this round
                best_response = max(item["response"], key=lambda x: x["accuracy"])
                # if the accuracy is larger than 0.9, stop
                if best_response["accuracy"] >= 0.9:
                    item["selected_response"] = best_response["response"]
                else:
                    item["response_candidates"].append(best_response)
                # if this is the last round, select the best response
                if round_index == self.refine_round - 1:
                    item["selected_response"] = max(item["response"], key=lambda x: x["accuracy"])["response"]

            # write result after each round
            with open(output_path, "w", encoding='utf-8') as f:
                for item in processed_items:
                    f.write(json.dumps(item) + "\n")

    def eval_srr(self, data_path, round=0):
        """
        Evaluate the responses generated by SRR on specified round.
        """
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        python_executor = PythonExecutor()
        error_count = 0
        for item in tqdm(data, desc=f"Evaluating SRR responses in round {round}"):
            max_round = max([resp["round"] for resp in item["response"]])
            if round == max_round:
                best_response = item["selected_response"]
            else:
                response_candidates =[resp for resp in item["response"] if resp["round"] <= round]
                best_response = max(response_candidates, key=lambda x: x["accuracy"])
                best_response = best_response["response"]
            try:
                function_string = extract_function(best_response)
                function_name = extract_function_names(function_string)[0]
                for test_example in item["test"]:
                    input_data = self.get_input(test_example)
                    test_example["res"] = python_executor.execute(
                        f"{function_string}\n",
                        "x=" + input_data,
                        f"result = {function_name}(x)"
                    )
            except Exception as e:
                error_count += 1
                for test_example in item["test"]:
                    test_example["res"] = "error_occurred"
        item_correct = 0
        item_count = 0
        task_correct = 0
        task_count = 0
        for item in data:
            all_correct = True
            for test_example in item["test"]:
                item_count += 1
                if self.check_correctness(test_example["res"], test_example["output"]):
                    item_correct += 1
                else:
                    all_correct = False
            if all_correct:
                task_correct += 1
            task_count += 1
        print("error count: ", error_count)
        item_accuracy = item_correct / item_count
        task_accuracy = task_correct / task_count
        print(f"Task accuracy in round {round}: {task_accuracy}")
        return task_accuracy

