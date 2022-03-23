from models.knowledge_harvester import KnowledgeHarvester


def main():
    knowledge_harvester = KnowledgeHarvester(model_name='roberta-large')

    # knowledge_harvester.init_prompts(prompts=[
    #     '<ENT1>\'s capital is <ENT0>',
    #     '<ENT0> is <ENT1>\'s capital',
    #     '<ENT0> is the capital of <ENT1> and one of the most populous cities in the world',
    #     'the capital of <ENT1> is <ENT0>',
    #     '<ENT1>\'s capital city is <ENT0>'])

    knowledge_harvester.init_prompts(prompts=[
        '<ENT0> is part of <ENT1>',
        '<ENT0> belongs to <ENT1>'])

    knowledge_harvester.update_ent_tuples()


if __name__ == '__main__':
    main()