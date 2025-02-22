#!/bin/bash

python run_task.py --task list_function\
    --model deepseek \
    --data_path datasets/list_functions/list_functions.jsonl \
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/list_function \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.0\
    --mode srr \
    --position -1 \
    --do_eval


python run_task.py --task list_function\
    --model deepseek \
    --data_path datasets/list_functions/list_functions.jsonl \
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/list_function \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.1\
    --mode srr \
    --position -1 \
    --do_eval


python run_task.py --task crypto\
    --model deepseek \
    --data_path datasets/crypto/atbash.jsonl \
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/crypto \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.0\
    --mode srr \
    --task_type atbash \
    --position -1 \
    --do_eval

python run_task.py --task crypto\
    --model deepseek \
    --data_path datasets/crypto/atbash.jsonl \
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/crypto \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.1\
    --mode srr \
    --task_type atbash \
    --position -1 \
    --do_eval

python run_task.py --task crypto\
    --model deepseek \
    --data_path datasets/crypto/caesar.jsonl \
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/crypto \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.0\
    --mode srr \
    --task_type caesar \
    --position -1 \
    --do_eval


python run_task.py --task crypto\
    --model deepseek \
    --data_path datasets/crypto/caesar.jsonl \
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/crypto \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.1\
    --mode srr \
    --task_type caesar \
    --position -1 \
    --do_eval


python run_task.py --task crypto\
    --model deepseek \
    --data_path datasets/crypto/keyboard.jsonl \
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/crypto \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.0\
    --mode srr \
    --task_type keyboard \
    --position -1 \
    --do_eval


python run_task.py --task crypto\
    --model deepseek \
    --data_path datasets/crypto/keyboard.jsonl \
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/crypto \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.1\
    --mode srr \
    --task_type keyboard \
    --position -1 \
    --do_eval


python run_task.py --task arithmetic \
    --model deepseek\
    --data_path datasets/arithmetic/arithmetic_base9_digits2.jsonl\
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/arithmetic \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.0\
    --mode srr \
    --position -1 \
    --task_type 9 \
    --do_eval

python run_task.py --task arithmetic \
    --model deepseek\
    --data_path datasets/arithmetic/arithmetic_base9_digits2.jsonl\
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/arithmetic \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.1\
    --mode srr \
    --position -1 \
    --task_type 9 \
    --do_eval


python run_task.py --task arithmetic \
    --model deepseek\
    --data_path datasets/arithmetic/arithmetic_base8_digits2.jsonl\
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/arithmetic \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.0\
    --mode srr \
    --position -1 \
    --task_type 8 \
    --do_eval

python run_task.py --task arithmetic \
    --model deepseek\
    --data_path datasets/arithmetic/arithmetic_base8_digits2.jsonl\
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/arithmetic \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.1\
    --mode srr \
    --position -1 \
    --task_type 8 \
    --do_eval


python run_task.py --task arithmetic \
    --model deepseek\
    --data_path datasets/arithmetic/arithmetic_base7_digits2.jsonl\
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/arithmetic \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.0\
    --mode srr \
    --position -1 \
    --task_type 7 \
    --do_eval

python run_task.py --task arithmetic \
    --model deepseek\
    --data_path datasets/arithmetic/arithmetic_base7_digits2.jsonl\
    --config_path config/api_key.json\
    --output_path evaluation/deepseek/arithmetic \
    --n_train 10\
    --n_test 10\
    --ood_ratio 0.0\
    --noise_ratio 0.1\
    --mode srr \
    --position -1 \
    --task_type 7 \
    --do_eval