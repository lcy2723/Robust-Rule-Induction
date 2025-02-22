import json
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--clean_path", type=str, required=True)
    parser.add_argument("--noisy_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_args()


def compute_consistency_score(clean_path, noisy_path, data_path):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    clean_path = os.path.basename(clean_path)
    noisy_path = os.path.basename(noisy_path)
    clean_complete = []
    noisy_complete = []
    for item in data:
        if item["file_name"] == clean_path:
            clean_complete = item["completed"]
        if item["file_name"] == noisy_path:
            noisy_complete = item["completed"]
    both_right = 0
    both_wrong = 0
    clean_right_noisy_wrong = 0
    clean_wrong_noisy_right = 0
    for clean, noisy in zip(clean_complete, noisy_complete):
        if clean == 1 and noisy == 1:
            both_right += 1
        elif clean == 0 and noisy == 0:
            both_wrong += 1
        elif clean == 1 and noisy == 0:
            clean_right_noisy_wrong += 1
        elif clean == 0 and noisy == 1:
            clean_wrong_noisy_right += 1
    # consistency score
    consistency_score = (both_right + both_wrong) / len(clean_complete)
    print(f"Both right: {both_right}, Both wrong: {both_wrong}, Clean right Noisy wrong: {clean_right_noisy_wrong}, Clean wrong Noisy right: {clean_wrong_noisy_right}")
    print(f"Consistency score: {consistency_score}")
    with open(os.path.join(os.path.dirname(data_path), "consistency_score.jsonl"), "a+") as f:
        f.write(json.dumps({
            "clean_path": clean_path,
            "noisy_path": noisy_path,
            "consistency_score": consistency_score,
            "both_right": both_right,
            "both_wrong": both_wrong,
            "clean_right_noisy_wrong": clean_right_noisy_wrong,
            "clean_wrong_noisy_right": clean_wrong_noisy_right
        }) + "\n")


if __name__ == "__main__":
    args = parse_args()
    clean_path = args.clean_path
    noisy_path = args.noisy_path
    data_path = args.data_path
    compute_consistency_score(clean_path, noisy_path, data_path)
    