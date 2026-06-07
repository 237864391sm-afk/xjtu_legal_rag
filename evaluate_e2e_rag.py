# -*- coding: utf-8 -*-
"""
智能法律 RAG 系统 - 端到端防幻觉生成质量自动化评测脚本
功能：自动从本地读取测试集，进行 Baseline(无检索直答版) 与 RAG (智能去噪版) 的打分对比。
"""

import os
import json
import re
from openai import OpenAI

import config
from vector_store import LegalVectorStore
from router import LegalIntentRouter_ZeroShot, LegalIntentRouter_LLM, LegalIntentRouter_CustomBERT

# ==============================================================================
# 全局配置区
# ==============================================================================
JUDGE_MODEL = "deepseek-chat"  # 裁判大模型底座
SAMPLE_SIZE = 50  # 学术评测样本规模


def load_cases_from_csv(file_name, max_samples=50):
    """从项目 data 目录下健壮地读取测试集文本"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "data", file_name)

    if not os.path.exists(file_path):
        print(f"找不到测试集文件: {file_path}")
        return []

    texts = []
    print(f"正在从 {file_name} 中读取前 {max_samples} 条测试样本...")
    with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or 'text' in line or '---' in line or 'label' in line:
                continue
            line = line.strip('|').strip()

            if '|' in line:
                parts = line.rsplit('|', 1)
            elif ',' in line:
                parts = line.rsplit(',', 1)
            else:
                parts = [line]

            if len(parts) >= 1:
                query_text = parts[0].strip()
                if query_text:
                    texts.append(query_text)

            if len(texts) >= max_samples:
                break
    return texts


def get_baseline_answer(client, query):
    """无 RAG 的纯大模型直答 (Baseline)"""
    system_prompt = """你是一名严谨的中国执业律师。请仅依靠内部知识回答用户的法律咨询，不可使用搜索引擎。

    执业要求：在进行法律定性或提供建议时，必须精确引用具体的国家成文法名称及精确的数字条款序号（例如：《中华人民共和国刑法》第二百六十六条）。
    严禁使用模糊表述。若涉及罪名或责任，必须明确指出相关法律及条款号。"""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def get_rag_answer(client, query, router, vector_db):
    """有 RAG 的本系统级回答 (Ours)"""
    try:
        # 1. 必查法条库
        statute_evidences = vector_db.search(query, "statute", top_k=config.TOP_K_STATUTE)

        # 2. 路由与案例召回
        route_result = router.get_route(query)
        intent = route_result.get("intent", route_result.get("target_library_name", "未知"))
        score = route_result.get("score", route_result.get("confidence", 0.0))

        case_evidences = []
        if score < config.ROUTER_CONFIDENCE_THRESHOLD:
            case_evidences.extend(vector_db.search(query, "criminal", top_k=config.TOP_K_CRIMINAL))
            case_evidences.extend(vector_db.search(query, "civil", top_k=config.TOP_K_CIVIL))
        elif intent == "刑事犯罪案件" or intent == 0:
            case_evidences = vector_db.search(query, "criminal", top_k=config.TOP_K_CRIMINAL)
        elif intent == "民商事纠纷" or intent == 1:
            case_evidences = vector_db.search(query, "civil", top_k=config.TOP_K_CIVIL)
        else:
            case_evidences.extend(vector_db.search(query, "criminal", top_k=config.TOP_K_CRIMINAL))
            case_evidences.extend(vector_db.search(query, "civil", top_k=config.TOP_K_CIVIL))

        # 3. 组装证据块
        all_evidences = statute_evidences + case_evidences
        context_block = "\n".join([f"[证据{i + 1}] {doc['content'] if isinstance(doc, dict) else str(doc)}" for i, doc in
                                   enumerate(all_evidences)])

        # 4. 智能去噪 Prompt
        system_prompt = """你是一名中国执业律师。请阅读【参考证据】并结合【自身的专业法律知识】回答用户的【法律咨询】。

        核心执业原则：
        1. 智能去噪：参考证据由系统检索匹配，可能包含噪音。必须具备专业甄别能力，忽略并剔除与案情明显无关的证据。
        2. 知识兜底：若检索证据未完全覆盖案件关键定性，必须调用自身法律知识储备进行补充。
        3. 刑民交叉：遇到复合案件，必须全面指出刑事和民事双重责任，不可漏判。
        """

        user_prompt = f"【参考证据】:\n{context_block}\n\n【法律咨询】:\n{query}\n\n请给出全面、权威的法律分析报告："

        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def llm_as_a_judge(client, query, answer_baseline, answer_rag):
    """LLM-as-a-Judge 自动化盲评裁判"""
    judge_prompt = f"""
    你是一个严格的法官。请盲评两个 AI 系统对同一法律案件的解答。

    【用户案件】
    {query}

    【解答 A (Baseline)】
    {answer_baseline}

    【解答 B (RAG System)】
    {answer_rag}

    【裁判注意事项】
    解答 B 是基于垂直法律知识库的 RAG 系统。其引用的带有 [证据X] 标识的案例均来源于真实司法文书库。不可将解答 B 引用这些案例判定为“幻觉”。
    “法条幻觉”(SHR) 严格指代：凭空捏造公开成文法名称，或扭曲、错误引用具体的法条数字内容。

    请按以下三个维度打分，并严格输出 JSON 格式：
    1. SHR (法条幻觉): 是否凭空捏造了成文法名称或写错了具体条文序号？(0=无幻觉, 1=有幻觉)
    2. QC (定性完整度): 若为复合案件，是否全面指出了责任？(0=漏判, 1=完整)
    3. Score (综合专家评分): 1-5 整数分，5分为最优。
    4. Reasoning (判定理由): 简要说明打分理由。

    JSON 返回格式示例:
    {{"Baseline": {{"SHR": 1, "QC": 0, "Score": 2}}, "RAG": {{"SHR": 0, "QC": 1, "Score": 5}}, "Reasoning": "打分理由..."}}
    """
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1
        )
        raw_content = response.choices[0].message.content.strip()
        clean_json = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if clean_json:
            return json.loads(clean_json.group())
        return json.loads(raw_content)
    except Exception as e:
        print(f"   [裁判系统异常] 解析失败: {e}")
        return {"Baseline": {"SHR": 0, "QC": 0, "Score": 3}, "RAG": {"SHR": 0, "QC": 0, "Score": 3},
                "Reasoning": "解析错误"}


def main():
    print("======================================================================")
    print("智能法律 RAG 系统 - 端到端防幻觉生成学术评测")
    print("======================================================================\n")

    print("正在初始化底层系统组件...")
    try:
        if config.ACTIVE_ROUTER_TYPE == "LLM":
            router = LegalIntentRouter_LLM(api_key=config.DEEPSEEK_API_KEY)
        elif config.ACTIVE_ROUTER_TYPE == "CustomBERT":
            router = LegalIntentRouter_CustomBERT(
                model_path=config.CUSTOM_ROUTER_MODEL_DIR,
                tau=getattr(config, 'ROUTER_CONFIDENCE_THRESHOLD', 0.60),
                entropy_threshold=getattr(config, 'ROUTER_ENTROPY_THRESHOLD', 1.0)
            )
        else:
            router = LegalIntentRouter_ZeroShot(model_name=config.ROUTER_MODEL_NAME)

        vector_db = LegalVectorStore(config.DATA_DIR, model_name=config.EMBEDDING_MODEL_NAME)
        vector_db.load_or_build_index("statute")
        vector_db.load_or_build_index("civil")
        vector_db.load_or_build_index("criminal")
    except Exception as e:
        print(f"系统初始化失败: {e}")
        return

    queries = load_cases_from_csv("test_data_adversarial.csv", max_samples=SAMPLE_SIZE)
    if not queries:
        queries = load_cases_from_csv("test_data_standard.csv", max_samples=SAMPLE_SIZE)

    if not queries:
        print("错误：未能在 data 目录下找到有效的 CSV 测试集语料。")
        return

    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

    stats = {
        "Baseline": {"SHR_count": 0, "QC_count": 0, "Total_Score": 0},
        "RAG": {"SHR_count": 0, "QC_count": 0, "Total_Score": 0}
    }

    valid_counts = 0

    print(f"\n开始对 {len(queries)} 条样本执行端到端对比评测...")
    for idx, query in enumerate(queries):
        print(f"[{idx + 1}/{len(queries)}] 正在评测样本: {query[:25]}...")

        # 1. 跑 Baseline
        ans_base = get_baseline_answer(client, query)

        # 2. 跑 Ours RAG
        ans_rag = get_rag_answer(client, query, router, vector_db)

        if ans_rag.startswith("Error"):
            print(f"   [异常] RAG 系统错误:\n   {ans_rag}")
            continue

        # 3. 召唤盲评裁判
        score_dict = llm_as_a_judge(client, query, ans_base, ans_rag)

        reasoning = score_dict.get("Reasoning", "无理由")
        if score_dict.get("RAG", {}).get("SHR") == 1 or score_dict.get("RAG", {}).get("Score", 5) < 3:
            print(f"   [裁判低分理由]: {reasoning}")
        else:
            print(f"   [裁判评价]: {reasoning}")

        try:
            stats["Baseline"]["SHR_count"] += score_dict["Baseline"]["SHR"]
            stats["Baseline"]["QC_count"] += score_dict["Baseline"]["QC"]
            stats["Baseline"]["Total_Score"] += score_dict["Baseline"]["Score"]

            stats["RAG"]["SHR_count"] += score_dict["RAG"]["SHR"]
            stats["RAG"]["QC_count"] += score_dict["RAG"]["QC"]
            stats["RAG"]["Total_Score"] += score_dict["RAG"]["Score"]

            valid_counts += 1
        except KeyError:
            continue

    if valid_counts == 0:
        print("评测未能成功录入任何数据。")
        return

    base_shr = (stats["Baseline"]["SHR_count"] / valid_counts) * 100
    base_qc = (stats["Baseline"]["QC_count"] / valid_counts) * 100
    base_avg_score = stats["Baseline"]["Total_Score"] / valid_counts

    rag_shr = (stats["RAG"]["SHR_count"] / valid_counts) * 100
    rag_qc = (stats["RAG"]["QC_count"] / valid_counts) * 100
    rag_avg_score = stats["RAG"]["Total_Score"] / valid_counts

    print("\n===================== 端到端生成质量学术成绩单 =====================")
    format_row = "{:<25} | {:<15} | {:<15} | {:<15}"
    print(format_row.format("系统架构方案", "法条引用幻觉率(SHR)↓", "定性完整率(QC)↑", "专家综合评分(1-5)↑"))
    print("-" * 80)
    print(format_row.format("No-RAG 直答基准 (Baseline)", f"{base_shr:.1f}%", f"{base_qc:.1f}%", f"{base_avg_score:.2f}"))
    print(format_row.format("本研究全链路 RAG 系统 (Ours)", f"{rag_shr:.1f}%", f"{rag_qc:.1f}%", f"{rag_avg_score:.2f}"))
    print("=======================================================================")
    print(f"注：本次盲评有效总样本数：{valid_counts} 条。")


if __name__ == "__main__":
    main()