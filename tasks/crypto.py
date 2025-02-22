import logging
import os
import json
from tqdm import tqdm
from tasks.base import Task
import random


class Crypto(Task):
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

    def get_input(self, item):
        return '"' + item["input"] + '"'

    @staticmethod
    def synthetic_data(data_path, size, cipher_type):
        from nltk.corpus import words
        # Casear cipher, Atbash cipher, keyboard cipher
        def caesar_cipher(text, shift=3):
            result = ""
            for i in range(len(text)):
                char = text[i]
                if char.isupper():
                    result += chr((ord(char) + shift - 65) % 26 + 65)
                else:
                    result += chr((ord(char) + shift - 97) % 26 + 97)
            return result
        
        def atbash_cipher(text, shift=1):
            result = ""
            for i in range(len(text)):
                char = text[i]
                if char.isupper():
                    result += chr(90 - (ord(char) - 65))
                else:
                    result += chr(122 - (ord(char) - 97))
            return result
        
        def keyboard_cipher(text, shift=1):
            keyboard = "QWERTYUIOPASDFGHJKLZXCVBNM"
            original = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            result = ""
            for i in range(len(text)):
                char = text[i]
                if char.isupper():
                    result += keyboard[(original.index(char)) % 26]
                else:
                    result += keyboard[(original.index(char.upper())) % 26].lower()
            return result

        if cipher_type == "caesar":
            cipher = caesar_cipher
        elif cipher_type == "atbash":
            cipher = atbash_cipher
        elif cipher_type == "keyboard":
            cipher = keyboard_cipher
        else:
            raise ValueError("Unknown cipher type")

        data = []    
        plain_texts = [w for w in words.words() if len(w) > 3 and len(w) <= 10]
        for _ in tqdm(range(size), desc="Generating data"):
            item = {"train": {"normal": [], "ood": [], "noise": []}, "test": []}
            word_list = random.sample(plain_texts, 30)
            word_shift = random.randint(1, 5)
            item["train"]["normal"] = [{"input": word, "output": cipher(word, word_shift)} for word in word_list[:10]]
            for i in range(10, 15):
                # shuffle the characters in the word
                word = word_list[i]
                word = list(word)
                random.shuffle(word)
                word = "".join(word)
                item["train"]["ood"].append({"input": word, "output": cipher(word, word_shift)})
            for i in range(15, 20):
                # add noise
                cipher_text = cipher(word_list[i])
                # replace characters with random characters
                word = list(cipher_text)
                modified_len = random.randint(1, len(word))
                for _ in range(modified_len):
                    idx = random.randint(0, len(word)-1)
                    error_char = random.choice("abcdefghijklmnopqrstuvwxyz")
                    while error_char == word[idx]:
                        error_char = random.choice("abcdefghijklmnopqrstuvwxyz")
                    word[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
                word = "".join(word)
                item["train"]["noise"].append({"input": word_list[i], "output": word})
            item["test"] = [{"input": word, "output": cipher(word, word_shift)} for word in word_list[20:]]
            data.append(item)
        with open(os.path.join(data_path, f"{cipher_type}.jsonl"), "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    data_path = "../datasets/crypto"
    Crypto.synthetic_data(data_path, 250, "caesar")
    Crypto.synthetic_data(data_path, 250, "atbash")
    Crypto.synthetic_data(data_path, 250, "keyboard")