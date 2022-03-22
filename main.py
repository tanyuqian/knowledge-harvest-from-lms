from models.knowledge_harvester import KnowledgeHarvester


def main():
    knowledge_harvester = KnowledgeHarvester(model_name='roberta-large')

    knowledge_harvester.init_prompts(prompts=['<ENT0> is part of <ENT1>'])
    knowledge_harvester.harvest()


if __name__ == '__main__':
    main()