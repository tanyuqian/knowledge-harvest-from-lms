# methods -------------------------------------------------
CUDA_VISIBLE_DEVICES=2 python main.py --rel_set humaneval --use_init_prompts --model_name roberta-large # running
CUDA_VISIBLE_DEVICES=5 python main.py --rel_set humaneval --n_prompts 1 --model_name roberta-large # running
# CUDA_VISIBLE_DEVICES=2 python main.py --rel_set humaneval --n_prompts 5 --model_name roberta-large # already done
CUDA_VISIBLE_DEVICES=2 python main.py --rel_set humaneval --use_auto_prompts --model_name roberta-large # running

CUDA_VISIBLE_DEVICES=2 python main.py --rel_set humaneval --use_lpaqa --model_name roberta-large # not implemented yet
CUDA_VISIBLE_DEVICES=2 python main.py --rel_set humaneval --use_human --model_name roberta-large # not implemented yet

# models ---------------------------------------------------

CUDA_VISIBLE_DEVICES=5 python main.py --rel_set humaneval --n_prompts 5 --model_name roberta-large # running 
CUDA_VISIBLE_DEVICES=3 python main.py --rel_set humaneval --n_prompts 5 --model_name bert-large-cased # running 
CUDA_VISIBLE_DEVICES=2 python main.py --rel_set humaneval --n_prompts 5 --model_name roberta-base # running 
CUDA_VISIBLE_DEVICES=2 python main.py --rel_set humaneval --n_prompts 5 --model_name bert-base-cased # running
CUDA_VISIBLE_DEVICES=2 python main.py --rel_set humaneval --n_prompts 5 --model_name distilbert-base-cased # running 
# (not in paper) CUDA_VISIBLE_DEVICES=3 python main.py --rel_set humaneval --n_prompts 5 --model_name bert-large-uncased