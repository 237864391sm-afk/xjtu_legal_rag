import os
import json
import config
from vector_store import LegalVectorStore
from generator import LegalLLMGenerator
from router import LegalIntentRouter_ZeroShot, LegalIntentRouter_LLM, LegalIntentRouter_CustomBERT


def main():
    print("==================================================")
    print("智能法律 RAG 系统 (数据透传版)")
    print("==================================================\n")

    try:
        # 1. 动态加载意图路由器
        if config.ACTIVE_ROUTER_TYPE == "LLM":
            router = LegalIntentRouter_LLM(api_key=config.DEEPSEEK_API_KEY)
        elif config.ACTIVE_ROUTER_TYPE == "CustomBERT":
            router = LegalIntentRouter_CustomBERT(model_path=config.CUSTOM_ROUTER_MODEL_DIR)
        else:
            router = LegalIntentRouter_ZeroShot(model_name=config.ROUTER_MODEL_NAME)

        # 2. 初始化生成器和检索引擎
        generator = LegalLLMGenerator(api_key=config.DEEPSEEK_API_KEY, model_name=config.LLM_MODEL_NAME)
        vector_db = LegalVectorStore(config.DATA_DIR, model_name=config.EMBEDDING_MODEL_NAME)

        # 3. 加载底层向量索引库
        vector_db.load_or_build_index("statute")
        vector_db.load_or_build_index("civil")
        vector_db.load_or_build_index("criminal")

    except Exception as e:
        print(f"\n[错误] 系统初始化失败: {e}")
        return

    print("\n[系统就绪] 输入问题即可开始分析。")

    # 获取信息熵阈值，默认 0.80
    entropy_threshold = getattr(config, "ROUTER_ENTROPY_THRESHOLD", 0.80)

    while True:
        query = input("\n请输入您的法律问题：\n> ")
        if query.lower() in ['quit', 'exit', '退出']:
            break
        if not query.strip():
            continue

        # 初始化前端数据透传包
        ui_data_dump = {
            "query": query,
            "router": {},
            "retrieval": [],
            "llm_report": {}
        }

        print("\n正在进行语义路由与检索...")

        # 检索阶段 1: 必查法条库 (不受路由影响，作为基础召回)
        statute_evidences = vector_db.search(query, "statute", top_k=config.TOP_K_STATUTE)

        # 检索阶段 2: 智能路由判断与兜底分发
        route_result = router.get_route(query)

        intent = route_result.get("intent", route_result.get("target_library_name", "未知"))
        score = route_result.get("score", route_result.get("confidence", 0.0))
        entropy = route_result.get("entropy", 0.0)

        print(f"   -> [网关日志] 最大置信度 C_max: {score:.4f} | 信息熵 H: {entropy:.4f}")

        case_evidences = []
        route_decision_msg = ""

        # 双重阈值联合拦截 (应对刑民交叉与对抗样本)
        is_low_confidence = score < config.ROUTER_CONFIDENCE_THRESHOLD
        is_high_entropy = entropy > entropy_threshold

        if is_low_confidence or is_high_entropy or "兜底" in str(intent):
            if is_high_entropy:
                route_decision_msg = f"捕获高熵状态 (H={entropy:.4f})，判定为[刑民交叉]或高噪音，触发[全域无损兜底搜索]"
            else:
                route_decision_msg = f"置信度偏低 (C_max={score:.4f})，意图模糊，触发[全域无损兜底搜索]"
            print(f"   -> [分流决策] {route_decision_msg}")

            case_evidences.extend(vector_db.search(query, "criminal", top_k=config.TOP_K_CRIMINAL))
            case_evidences.extend(vector_db.search(query, "civil", top_k=config.TOP_K_CIVIL))

        elif intent == "刑事犯罪案件" or intent == 0:
            route_decision_msg = "意图清晰，单路直连: 检索刑事案例库"
            print(f"   -> [分流决策] {route_decision_msg}")
            case_evidences = vector_db.search(query, "criminal", top_k=config.TOP_K_CRIMINAL)

        elif intent == "民商事纠纷" or intent == 1:
            route_decision_msg = "意图清晰，单路直连: 检索民商事案例库"
            print(f"   -> [分流决策] {route_decision_msg}")
            case_evidences = vector_db.search(query, "civil", top_k=config.TOP_K_CIVIL)

        else:
            route_decision_msg = f"命中未知领域 ({intent})，安全降级至[全域兜底搜索]"
            print(f"   -> [分流决策] {route_decision_msg}")
            case_evidences.extend(vector_db.search(query, "criminal", top_k=config.TOP_K_CRIMINAL))
            case_evidences.extend(vector_db.search(query, "civil", top_k=config.TOP_K_CIVIL))

        # 记录路由状态
        ui_data_dump["router"] = {
            "confidence": round(score, 4),
            "entropy": round(entropy, 4),
            "intent": intent,
            "decision_msg": route_decision_msg,
            "is_intercepted": is_low_confidence or is_high_entropy or "兜底" in str(intent)
        }

        # 组合全部参考证据
        all_evidences = statute_evidences + case_evidences

        print(f"\n已召回 {len(all_evidences)} 条参考依据：")
        print("-" * 50)
        for i, ev in enumerate(all_evidences):
            source_type = "法条" if ev['metadata'].get('type') == 'statute' else "案例"
            ev_score = ev.get('score', 0)
            print(f"[{i + 1}] {source_type} | ID: {ev.get('id', 'N/A')} | 相似度: {ev_score:.4f}")

            ui_data_dump["retrieval"].append({
                "index": i + 1,
                "source_type": source_type,
                "title": ev.get('id', 'N/A'),
                "score": round(ev_score, 4)
            })
        print("-" * 50)

        # 大模型生成与防幻觉推演
        print("\n引擎激活：正在生成结构化报告与去噪日志...\n")
        final_answer = generator.generate(query, all_evidences)

        try:
            if isinstance(final_answer, str):
                clean_json = final_answer.replace('```json', '').replace('```', '').strip()
                report = json.loads(clean_json)
            else:
                report = final_answer

            # 打印防幻觉日志
            print("=================[大模型防幻觉去噪日志]=================")
            print(f"   {report.get('去噪日志', '未生成去噪日志')}")
            print("========================================================\n")

            # 打印最终五维分析报告
            print("=================[智能法律分析报告]=================")
            print(f"[初步结论]\n   {report.get('初步结论', '未生成')}\n")
            print(f"[法律依据]\n   {report.get('法律依据', '未生成')}\n")
            print(f"[法理与事实分析]\n   {report.get('法理与事实分析', '未生成')}\n")
            print(f"[风险提示]\n   {report.get('风险提示', '未生成')}\n")
            print(f"[行动建议]\n   {report.get('行动建议', '未生成')}\n")
            print("====================================================\n")

            ui_data_dump["llm_report"] = report
            ui_data_dump["parse_status"] = "success"

        except Exception as parse_err:
            print(f"[解析错误] 结构化解析失败，返回原始文本：\n")
            print(final_answer)
            ui_data_dump["llm_report"] = {"raw_text": final_answer}
            ui_data_dump["parse_status"] = "failed"

        print("====================================================\n")

        # 前端 UI 数据包转储
        print("[前端 UI 数据源 JSON]")
        print("---UI_DATA_START---")
        print(json.dumps(ui_data_dump, ensure_ascii=False, indent=2))
        print("---UI_DATA_END---")
        print("====================================================\n")


if __name__ == "__main__":
    main()