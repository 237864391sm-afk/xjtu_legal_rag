# src/router.py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from openai import OpenAI


# ==============================================================================
# 1. 自研轻量化微调路由网关
# ==============================================================================
class LegalIntentRouter_CustomBERT:
    def __init__(self, model_path: str, device: str = None, tau: float = 0.60, entropy_threshold: float = 1.00):
        """
        :param model_path: 模型路径
        :param device: 运行设备
        :param tau: 置信度拦截阈值 (默认 0.60)
        :param entropy_threshold: 信息熵拦截阈值 (默认 1.00)
        """
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"[CustomBERT 路由] 正在加载微调模型至 {self.device}...")

        # 接收外部传入的阈值配置，实现逻辑解耦
        self.tau = tau
        self.entropy_threshold = entropy_threshold

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()  # 锁定评估模式
        except Exception as e:
            print(f"[CustomBERT 路由] 模型加载失败: {e}")
            self.model = None

        self.label_mapping = {
            0: "刑事犯罪案件",
            1: "民商事纠纷",
            2: "行政诉讼与国家赔偿"
        }

    def get_route(self, query: str) -> dict:
        if self.model is None:
            return {"intent": "未知", "score": 0.0, "entropy": 1.58}

        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 1. 获取三分类 Softmax 概率分布
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

        # 2. 计算最大置信度 C_max
        c_max = float(np.max(probabilities))
        pred_label_id = int(np.argmax(probabilities))

        # 3. 计算信息熵 H (引入 epsilon 防止 log2(0) 溢出)
        epsilon = 1e-9
        entropy = float(-np.sum([p * np.log2(p + epsilon) for p in probabilities if p > 0]))

        # =========================================================
        # 动态双重阈值联合拦截判定
        # 阈值 1: C_max >= self.tau
        # 阈值 2: H < self.entropy_threshold (3分类最大熵约为1.58)
        # =========================================================
        if c_max >= self.tau and entropy < self.entropy_threshold:
            # 满足置信度与信息熵阈值，允许单路直连
            intent = self.label_mapping.get(pred_label_id, "未知")
            return {"intent": intent, "score": c_max, "entropy": entropy}
        else:
            # 未满足阈值条件，触发全域兜底
            return {"intent": "未知_触发兜底", "score": c_max, "entropy": entropy}


# ==============================================================================
# 2. 零样本基座路由网关 (对比测试组，基于 NLI)
# ==============================================================================
class LegalIntentRouter_ZeroShot:
    def __init__(self, model_name: str, device: str = None):
        self.device = 0 if (device == 'cuda:0' or torch.cuda.is_available()) else -1
        print(f"[ZeroShot 路由] 正在加载零样本基座 {model_name}...")

        try:
            self.classifier = pipeline("zero-shot-classification", model=model_name, device=self.device)
        except Exception as e:
            print(f"[ZeroShot 路由] 模型加载失败: {e}")
            self.classifier = None

        self.candidate_labels = ["刑事犯罪案件", "民商事纠纷", "行政诉讼与国家赔偿"]

    def get_route(self, query: str) -> dict:
        if self.classifier is None:
            return {"intent": "未知", "score": 0.0, "entropy": 1.58}

        res = self.classifier(query, self.candidate_labels)
        intent = res['labels'][0]
        scores = res['scores']

        c_max = float(scores[0])

        epsilon = 1e-9
        entropy = float(-np.sum([p * np.log2(p + epsilon) for p in scores if p > 0]))

        return {
            "intent": intent,
            "score": c_max,
            "entropy": entropy
        }


# ==============================================================================
# 3. 云端大模型自回归路由网关 (API 方案)
# ==============================================================================
class LegalIntentRouter_LLM:
    def __init__(self, api_key: str):
        print("[LLM 路由] 正在初始化 API 云端路由器...")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

        self.system_prompt = (
            "你是一个法律意图分类器。请严格根据用户的提问，输出且仅输出以下三个分类之一：\n"
            "刑事犯罪案件\n"
            "民商事纠纷\n"
            "行政诉讼与国家赔偿"
        )

    def get_route(self, query: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,  # 强制确定性输出
                max_tokens=10
            )
            intent = response.choices[0].message.content.strip()

            # 大模型输出为生成式文本，非概率分布，固定置信度与信息熵
            return {
                "intent": intent,
                "score": 0.99,
                "entropy": 0.0
            }
        except Exception as e:
            print(f"[LLM 路由] API 请求失败: {e}")
            return {"intent": "未知", "score": 0.0, "entropy": 1.58}