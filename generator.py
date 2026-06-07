import json
from openai import OpenAI

class LegalLLMGenerator:
    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        self.model_name = model_name

    def generate(self, query: str, context_docs: list) -> str:
        context_str = ""
        for i, doc in enumerate(context_docs):
            title = doc['metadata'].get('title', doc.get('id', '未知来源'))
            context_str += f"[证据 {i+1}] {title}\n内容摘要：{doc.get('content', '')}\n\n"

        # 结构化 Prompt 模板，配置 CoT 与 JSON 约束
        system_prompt = """你是一个严谨的中国法律 RAG 智能助手。现在你必须扮演“审查法官”的角色，对系统召回的检索片段与用户事实进行独立校验，打破对检索结果的盲从。

        【指令约束与思维链(CoT)推演】
        1. 事实提取：精准提取用户输入中的核心法律事实要素。
        2. 强相关性鉴别（Self-RAG）：仔细比对提供的参考法条/案例与用户事实，忽略所有虽然词汇相似但适用前提不符的证据。若所有提供的法条均不适用，请明确声明‘依据现有知识无法得出结论’，严禁自行捏造法条。
        3. 双向视角推演：如果案件存在“刑民交叉”风险，必须同时从“刑事犯罪”与“民事违约/侵权”两个切面展开平行分析。
        4. 法律三段论：按照“提取案件事实要素 -> 匹配去噪后的法条 -> 推导定性结果”的步骤输出分析。
        5. 法条补充：你可以引用已经验证过在当时有效的明确法条

        【强制输出格式】
       你必须输出且仅输出一个合法的 JSON 对象，包含以下 6 个字段（第1个字段用于系统后台监控，后5个字段用于生成面向用户的报告）：
{
    "去噪日志": "明确指出提供的证据中，哪些被采纳了？哪些因为适用前提不符被丢弃/否决了？",
    "初步结论": "案件的定性结果（如'属于民事违约'或'涉嫌诈骗犯罪'）。",
    "法律依据": "依据法条与参考案例。",
    "法理与事实分析": "根据法理与事实进行案件分析，辅助用户确认逻辑链是否完整。",
    "风险提示": "基于用户描述，指出当前证据链中缺失的关键事实或潜在的法律风险。",
    "行动建议": "为用户提供合法、实用的下一步操作指引。"
}"""
        user_prompt = f"【检索证据】：\n{context_str}\n\n【用户咨询】：{query}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, # 约束输出为 JSON 模式
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            # 异常处理：保持 JSON 结构输出以保证下游解析稳定
            return json.dumps({
                "去噪日志": "未生成",
                "初步结论": "生成失败",
                "法律依据": "无",
                "法理与事实分析": f"调用大模型 API 发生错误：{str(e)}",
                "风险提示": "系统异常",
                "行动建议": "请检查网络或 API 配置"
            }, ensure_ascii=False)