"""
快速检索示例 - 基于 FlagEmbedding
Quick Retrieval Demo using FlagEmbedding

这个脚本演示如何使用 FlagEmbedding 快速构建一个检索系统。
This script demonstrates how to quickly build a retrieval system using FlagEmbedding.

使用方法 / Usage:
    python quick_retrieval_demo.py
"""

import numpy as np
from FlagEmbedding import FlagModel


def cosine_similarity(a, b):
    """计算余弦相似度 / Compute cosine similarity"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    print("=" * 60)
    print("FlagEmbedding 快速检索示例")
    print("FlagEmbedding Quick Retrieval Demo")
    print("=" * 60)

    # 1. 加载模型 / Load model
    print("\n[1] 加载模型 / Loading model...")
    model_name = "BAAI/bge-small-zh-v1.5"  # 使用中文小模型，速度快
    model = FlagModel(model_name, use_fp16=True)
    print(f"✓ 模型加载完成: {model_name}")

    # 2. 准备文档库 / Prepare document corpus
    print("\n[2] 准备文档库 / Preparing document corpus...")
    documents = [
        "北京是中国的首都，有着悠久的历史和丰富的文化遗产。",
        "上海是中国最大的城市，也是重要的金融中心。",
        "深圳是中国的科技创新中心，拥有众多高科技企业。",
        "杭州以西湖闻名，也是电子商务的重要基地。",
        "成都是四川省的省会，以美食和熊猫而著称。",
        "广州是中国南部的商贸和交通枢纽。",
        "南京是历史名城，曾为多朝古都。",
        "西安以兵马俑和悠久历史闻名。",
        "重庆以山城地形和火锅文化著称。",
        "天津拥有重要的港口和工业基础。",
        "Python是一种广泛使用的高级编程语言。",
        "机器学习是人工智能的一个重要分支。",
        "深度学习在图像识别和自然语言处理中表现出色。",
        "向量检索是现代搜索引擎的核心技术之一。",
        "大语言模型正在改变人机交互的方式。",
        "区块链技术在分布式账本和加密货币中被广泛使用。",
        "量子计算仍处于早期阶段，但有潜力加速特定类型计算任务。",
        "生态保护是实现可持续发展的重要组成部分。",
        "篮球是一项全球流行的运动项目，拥有广泛的关注者。",
        "古典音乐对西方音乐传统有深远影响。",
        "太空探索推动了航天、材料与通信技术的发展。",
        "可再生能源（如风能与太阳能）正在逐步替代化石燃料。",
        "统计学是数据科学的重要基础学科。",
        "自然语言处理关注让计算机理解和生成自然语言。",
        "计算机视觉使机器能够“看见”并理解图像内容。",
        "无人驾驶技术融合传感、控制与AI算法。",
    ]
    print(f"✓ 文档数量: {len(documents)}")

    # 3. 对文档进行编码 / Encode documents
    print("\n[3] 对文档进行编码 / Encoding documents...")
    doc_embeddings = model.encode(documents)
    print(f"✓ 编码完成，向量维度: {doc_embeddings.shape}")

    # 4. 执行检索 / Perform retrieval
    print("\n[4] 执行检索 / Performing retrieval...")
    print("-" * 60)

    # 查询示例 / Query examples
    queries = [
        "中国的首都在哪里？",
        "哪个城市是科技中心？",
        "什么是深度学习？",
        "有哪些可再生能源类型？",
        "量子计算的潜力是什么？",
        "请介绍有关自然语言处理的内容。",
        "推荐与机器学习相关的入门书籍或主题。",
        "哪个城市以美食和熊猫而著称？",
        "什么是向量检索？",
        "介绍一下太空探索的意义。",
    ]

    for query in queries:
        print(f"\n查询 / Query: {query}")

        # 对查询进行编码 / Encode query
        query_embedding = model.encode(query)

        # 计算相似度 / Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim = cosine_similarity(query_embedding, doc_emb)
            similarities.append((i, sim, documents[i]))

        # 排序并获取 Top-k / Sort and get Top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = 5

        print(f"\nTop-{top_k} 相关文档 / Top-{top_k} relevant documents:")
        for rank, (doc_id, score, doc_text) in enumerate(similarities[:top_k], 1):
            print(f"  {rank}. [相似度: {score:.4f}] {doc_text}")

    print("\n" + "=" * 60)
    print("检索完成！/ Retrieval completed!")
    print("=" * 60)

    # 5. 交互式检索 / Interactive retrieval
    print("\n[5] 交互式检索模式 / Interactive retrieval mode")
    print("输入您的查询（输入 'quit' 退出）/ Enter your query (type 'quit' to exit):")

    while True:
        try:
            user_query = input("\n> ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("退出程序 / Exiting...")
                break

            if not user_query:
                continue

            # 执行检索
            query_embedding = model.encode(user_query)
            similarities = []
            for i, doc_emb in enumerate(doc_embeddings):
                sim = cosine_similarity(query_embedding, doc_emb)
                similarities.append((i, sim, documents[i]))

            similarities.sort(key=lambda x: x[1], reverse=True)

            print(f"\nTop-3 结果 / Top-3 results:")
            for rank, (doc_id, score, doc_text) in enumerate(similarities[:3], 1):
                print(f"  {rank}. [相似度: {score:.4f}] {doc_text}")

        except KeyboardInterrupt:
            print("\n\n退出程序 / Exiting...")
            break
        except Exception as e:
            print(f"错误 / Error: {e}")


if __name__ == "__main__":
    main()
