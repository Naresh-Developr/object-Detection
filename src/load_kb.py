def loadKnowledgeBase(filePath):
    knowledge_base = {}
    with open(filePath, 'r') as file:
        for line in file:
            item, count = line.strip().split(',')
            knowledge_base[item.strip()] = int(count.strip())
    return knowledge_base
