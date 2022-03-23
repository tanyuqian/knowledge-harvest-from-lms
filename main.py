from models.knowledge_harvester import KnowledgeHarvester


def main():
    # knowledge_harvester = KnowledgeHarvester(model_name='roberta-large')
    #
    # knowledge_harvester.init_prompts(prompts=['<ENT0> is part of <ENT1>'])
    # knowledge_harvester.harvest()

    from models.language_model_wrapper import LanguageModelWrapper

    lm = LanguageModelWrapper(model_name='roberta-large')
    print(lm.get_mask_spans(prompt='<ENT0> is the capital of <ENT1>',
                      ent_tuple=['moscow', 'russia']))


if __name__ == '__main__':
    main()