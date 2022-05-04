CUDA_VISIBLE_DEVICES=7 python pr_scoring.py --rel_set lama --model roberta-large # already done
CUDA_VISIBLE_DEVICES=0 python pr_scoring.py --rel_set lama --model roberta-base --settings 5 # running
CUDA_VISIBLE_DEVICES=0 python pr_scoring.py --rel_set lama --model bert-large-cased --settings 5  # running

CUDA_VISIBLE_DEVICES=4 python pr_scoring.py --rel_set lama --model distilbert-base-cased --settings 5 # running
# CUDA_VISIBLE_DEVICES=7 python pr_scoring.py --rel_set lama --model bert-large-uncased

CUDA_VISIBLE_DEVICES=1 python pr_scoring.py --rel_set conceptnet --model roberta-large # need re-run
CUDA_VISIBLE_DEVICES=7 python pr_scoring.py --rel_set conceptnet --model roberta-base --settings 5 # running 
CUDA_VISIBLE_DEVICES=7 python pr_scoring.py --rel_set conceptnet --model bert-large-cased --settings 5 # running

CUDA_VISIBLE_DEVICES=4 python pr_scoring.py --rel_set conceptnet --model distilbert-base-cased --settings 5 # running
# CUDA_VISIBLE_DEVICES=7 python pr_scoring.py --rel_set conceptnet --model bert-large-uncased
