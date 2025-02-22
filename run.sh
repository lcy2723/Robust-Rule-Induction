#!/bin/bash

python run_task.py --task list_function\
    --model deepseek \
    --data_path datasets/list_functions/list_functions.jsonl \
    --config_path config/api_key.json\
    --output_path outputs \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.1\
    --mode srr \
    --position -1 \
    --do_eval