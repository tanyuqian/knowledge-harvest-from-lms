import fire

from models.knowledge_harvester import KnowledgeHarvester

from data_utils.ckbc import CKBC
from data_utils.data_utils import conceptnet_relation_init_prompts

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
import os

def get_pr_scores(harvester, testset, rel, prompts, weights):
    harvester.clear()
    harvester.init_prompts(prompts)
    weighted_ent_tuples = harvester.get_ent_tuples_weight(
            testset.get_ent_tuples(rel=rel), weights=weights)
    y_true, scores = [], []
    for ent_tuple, weight in weighted_ent_tuples:
        label = testset.get_label(rel=rel, ent_tuple=ent_tuple)
        y_true.append(label)
        scores.append(weight)
    scores_labels = list(zip(scores, y_true))
    scores_labels.sort(key=lambda x: x[0], reverse=True)
    precision, recall = [], []
    tp, p, t = 0, 0, sum(y_true)
    for score, label in scores_labels:
        p += 1
        tp += label
        precision.append(tp / p)
        recall.append(tp / t)
    return precision, recall

def main():
    test_file = "conceptnet_high_quality.txt"
    # weights = (.25, .25, 1)
    weights = (0, 1, 0)
    # test_file = 'test.txt'
    ckbc = CKBC(test_file)
    knowledge_harvester = KnowledgeHarvester(
        model_name='roberta-large', max_n_ent_tuples=None)
    save_dir = f'ft_reverse_0412_neww_ckbc_{test_file.split(".")[0]}_{weights}_curves'
    os.makedirs(save_dir, exist_ok=True)
    # target_rel = ["CapableOf"]
    target_rel = []
    for relation, init_prompts in conceptnet_relation_init_prompts.items():
        if relation not in ckbc._ent_tuples:
            continue
        if len(target_rel) > 0 and relation not in target_rel:
            continue
        n_tuples = len(ckbc.get_ent_tuples(rel=relation))
        
        for n_prompts in [1, 2, 3]:
            knowledge_harvester.clear()
            prompts = init_prompts
            # prompts = prompts[::-1][:n_prompts]
            precision, recall = get_pr_scores(
                knowledge_harvester, ckbc, relation, prompts, weights)
            # precision, recall, _ = precision_recall_curve(y_true, scores)
            # to aggregate the scores, the precision and recall should be of the same lengths
            # but the function from sklearn return vectors of different lengths

            plt.plot(recall, precision, label=f'{n_prompts}prompts')
        
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.legend()
        plt.title(f"{relation}: {n_tuples} tuples")
        plt.savefig(f"{save_dir}/{relation}.png")
        plt.figure().clear()

if __name__ == '__main__':
    fire.Fire(main)
