import os
import json
import math
from tqdm import tqdm
import config
from vector_store import LegalVectorStore


def calculate_ndcg(retrieved_scores, ground_truth_scores, k):
    """计算 NDCG@K 指标"""

    def dcg(scores):
        return sum([s / math.log2(i + 2) for i, s in enumerate(scores[:k])])

    ideal_scores = sorted(ground_truth_scores, reverse=True)
    idcg = dcg(ideal_scores)

    if idcg == 0:
        return 0.0
    return dcg(retrieved_scores) / idcg


def calculate_recall(retrieved_docs, ground_truth_docs, k):
    """计算 Recall@K 指标"""
    retrieved_k = retrieved_docs[:k]
    hits = sum(1 for doc in retrieved_k if doc in ground_truth_docs)
    total_relevant = len(ground_truth_docs)

    if total_relevant == 0:
        return 0.0
    return hits / total_relevant


def main():
    print("==================================================")
    print("法律 RAG 检索层自动化评测 (LeCaRDv2)")
    print("==================================================\n")

    # 1. 加载测试集与标注数据
    queries_path = os.path.join(config.PROCESSED_DIR, "test_queries.json")
    qrels_path = os.path.join(config.PROCESSED_DIR, "qrels_test.json")

    if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
        print("评测数据缺失，请检查 data/processed 目录。")
        return

    with open(queries_path, 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
    with open(qrels_path, 'r', encoding='utf-8') as f:
        qrels = json.load(f)

    # 2. 初始化向量检索引擎 (针对 LeCaRDv2 加载刑事案例库)
    vector_db = LegalVectorStore(config.DATA_DIR, model_name=config.EMBEDDING_MODEL_NAME)
    vector_db.load_or_build_index("criminal")

    # 评测参数配置
    K_LIST = [5, 10, 20]
    metrics = {k: {'ndcg': 0.0, 'recall': 0.0} for k in K_LIST}
    valid_queries = 0

    print(f"\n开始评测 {len(test_queries)} 条查询样本...\n")

    # 3. 执行检索测试
    # 注: 当前取前 100 条进行快速验证，全量测试请移除 [:100] 切片
    for item in tqdm(test_queries[:100]):
        qid = str(item['qid'])
        query_text = item['query']

        if qid not in qrels:
            continue

        ground_truth_dict = qrels[qid]
        ground_truth_docs = list(ground_truth_dict.keys())
        ground_truth_scores = list(ground_truth_dict.values())

        # 按照最大 K 值进行检索以满足后续截断计算需求
        max_k = max(K_LIST)
        results = vector_db.search(query_text, "criminal", top_k=max_k)
        retrieved_docs = [res['id'] for res in results]

        # 映射检索结果的相关性得分，未命中标准答案则记为 0 分
        retrieved_scores = [ground_truth_dict.get(doc, 0) for doc in retrieved_docs]

        # 累计指标分数
        for k in K_LIST:
            metrics[k]['ndcg'] += calculate_ndcg(retrieved_scores, ground_truth_scores, k)
            metrics[k]['recall'] += calculate_recall(retrieved_docs, ground_truth_docs, k)

        valid_queries += 1

    # 4. 输出评测报告
    print("\n\n============ 检索层评测报告 ============")
    print(f"评测基座模型: {config.EMBEDDING_MODEL_NAME}")
    print(f"有效测试样本: {valid_queries} 条")
    print("-" * 40)
    for k in K_LIST:
        avg_ndcg = metrics[k]['ndcg'] / valid_queries
        avg_recall = metrics[k]['recall'] / valid_queries
        print(f"Top-{k} 指标:")
        print(f"   - NDCG@{k}:   {avg_ndcg:.4f}")
        print(f"   - Recall@{k}: {avg_recall:.4f}")
    print("========================================\n")


if __name__ == "__main__":
    main()