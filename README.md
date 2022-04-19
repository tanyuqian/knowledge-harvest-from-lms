# knowledge-harvest-from-lms

## Results

### precision-recall curves to evaluate scoring

PR curves can be found [here](results/curves_scoring_pr).

* ckbc can be treated as the upper bound.
* overall, init prompt < 1 prompt < (slightly) multiple prompts


### Output entity tuples

Output entity tuples can be found in ```results/outputs_*prompts/summary.txt```, like [results/outputs_20prompts/summary.txt](results/outputs_20prompts/summary.txt)

### CKBC scores of output entity tuples

CKBC score curves can be found [here](results/curves_outputs_ckbc).

in most of relations: init prompt < 1 prompt < 20 prompts

overall scores (first 100/1000 entity tuples):
``` 
output_dir: results/outputs_1prompts:
ckbc: 0.3602940108432876
ckbc acc: 0.3068421052631579
==================================================
output_dir: results/outputs_5prompts:
ckbc: 0.44737082379014453
ckbc acc: 0.4225
==================================================
output_dir: results/outputs_10prompts:
ckbc: 0.4501047574291428
ckbc acc: 0.42
==================================================
output_dir: results/outputs_20prompts:
ckbc: 0.43482570637417983
ckbc acc: 0.402
==================================================
output_dir: results/outputs_initprompt:
ckbc: 0.2418785971091492
ckbc acc: 0.182
==================================================
```

