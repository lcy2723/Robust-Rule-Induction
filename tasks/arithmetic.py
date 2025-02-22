import logging
import json
import random
import os
from tqdm import tqdm
from tasks.base import Task

class Arithmetic(Task):
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

    def check_correctness(self, res, output):
        if not isinstance(res, str):
            return str(res) == output
        return res == output

    def get_input(self, item):
        return '"' + item["input"] + '"'
        
    @staticmethod
    def synthetic_data(data_path, size, base_num, digits_num=2):
        def convert_to_nbase(num, base):
            res = ''
            while num > 0:
                if num % base >= 10:
                    res = chr(ord('A') + num % base - 10) + res
                else:
                    res = str(num % base) + res
                num //= base
            return res
        
        def addition(a, b, base):
            return convert_to_nbase(int(a, base) + int(b, base), base)
        
        def check_carry(a: str, b: str, base):
            for i in range(len(a)):
                if int(a[i], base) + int(b[i], base) >= base:
                    return True
            return False
        
        def generate_data(base, digits, n_items):
            # Generate data in list
            inside_data = []
            for _ in range(n_items):
                a = convert_to_nbase(random.randint(base**(digits-1)+1, base**digits-1), base)
                b = convert_to_nbase(random.randint(base**(digits-1)+1, base**digits-1), base)
                while True:
                    a = convert_to_nbase(random.randint(base**(digits-1)+1, base**digits-1), base)
                    b = convert_to_nbase(random.randint(base**(digits-1)+1, base**digits-1), base)
                    if check_carry(a, b, base):
                        break
                c = addition(a, b, base)
                inside_data.append({"input": a + " + " + b, "output": c})
            return inside_data
        
        data = []
        for _ in tqdm(range(size), desc="Generating data"):
            item = {"train": {"normal": [], "ood": [], "noise": []}, "test": []}
            item["train"]["normal"] = generate_data(base_num, digits_num, 10)
            item["train"]["ood"] = generate_data(base_num, digits_num+1, 5)
            item["train"]["noise"] = generate_data(10, digits_num, 5)
            item["test"] = generate_data(base_num, digits_num, 10)
            # check overlap in train and test
            for i in range(10):
                while item["test"][i] in item["train"]["normal"] or item["test"][i] in item["train"]["ood"] or item["test"][i] in item["train"]["noise"]:
                    item["test"][i] = generate_data(base_num, digits_num, 1)[0]
            data.append(item)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        output_path = os.path.join(data_path, f"arithmetic_base{base_num}_digits{digits_num}.jsonl")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    Arithmetic.synthetic_data("../datasets/arithmetic", 100, 7, 2)
    Arithmetic.synthetic_data("../datasets/arithmetic", 100, 8, 2)
    Arithmetic.synthetic_data("../datasets/arithmetic", 100, 9, 2)