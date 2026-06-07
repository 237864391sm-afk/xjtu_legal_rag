import os
import time
import pandas as pd
import torch
from transformers import pipeline
from openai import OpenAI

# ==========================================
# 1. API 配置
# ==========================================
DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def load_test_data(file_name):
    """加载并清洗测试集数据"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "data", file_name)

    if not os.path.exists(file_path):
        print(f"找不到测试集: {file_path}")
        return None

    texts, labels = [], []
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
                continue
            if len(parts) == 2:
                texts.append(parts[0].strip())
                labels.append(parts[1].strip())

    df = pd.DataFrame({'text': texts, 'label': labels})
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
    return df[df['label'] != -1]


def evaluate_local_model(model_name_or_path, model_type, test_data, device):
    """评测本地部署模型 (Model A, B, C)"""
    try:
        if model_type == "zero-shot":
            classifier = pipeline("zero-shot-classification", model=model_name_or_path, device=device)
            candidate_labels = ["刑事犯罪案件", "民商事纠纷", "行政诉讼与国家赔偿"]
            label_to_id = {"刑事犯罪案件": 0, "民商事纠纷": 1, "行政诉讼与国家赔偿": 2}
            _ = classifier("预热", candidate_labels=["测试"])
        else:
            classifier = pipeline("text-classification", model=model_name_or_path, device=device)
            _ = classifier("预热")
    except Exception as e:
        print(f"本地模型加载失败: {e}")
        return 0.0, 0.0

    correct = 0
    start_time = time.time()

    for _, row in test_data.iterrows():
        text = str(row['text'])
        true_label = int(row['label'])

        if model_type == "zero-shot":
            res = classifier(text, candidate_labels)
            pred_label = label_to_id.get(res['labels'][0], -1)
        else:
            res = classifier(text)[0]
            pred_val = str(res['label']).replace('LABEL_', '')
            try:
                pred_label = int(pred_val)
            except ValueError:
                label_to_id = {"刑事犯罪案件": 0, "民商事纠纷": 1, "行政诉讼与国家赔偿": 2}
                pred_label = label_to_id.get(pred_val, -1)

        if pred_label == true_label:
            correct += 1

    end_time = time.time()
    return (correct / len(test_data)) * 100, ((end_time - start_time) / len(test_data)) * 1000


def evaluate_api_model(test_data):
    """评测云端大模型 API (Model D)"""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_API_KEY_HERE":
        print("未配置有效的 DEEPSEEK_API_KEY，跳过 API 模型评测。")
        return 0.0, 0.0

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    correct = 0
    start_time = time.time()

    system_prompt = (
        "你是一个法律意图分类器。请对用户的法律咨询进行分类。\n"
        "必须且只能输出以下三个数字之一，不要返回任何其他文字：\n"
        "0: 刑事犯罪案件\n"
        "1: 民商事纠纷\n"
        "2: 行政诉讼与国家赔偿"
    )

    for _, row in test_data.iterrows():
        text = str(row['text'])
        true_label = int(row['label'])

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                max_tokens=5
            )
            pred_val = response.choices[0].message.content.strip()
            # 提取可能包含的数字
            pred_label = int(''.join(filter(str.isdigit, pred_val)))
        except Exception as e:
            pred_label = -1

        if pred_label == true_label:
            correct += 1

    end_time = time.time()
    return (correct / len(test_data)) * 100, ((end_time - start_time) / len(test_data)) * 1000


def main():
    print("==============================================================")
    print("智能法律路由系统 - 多模型端到端评测")
    print("==============================================================\n")

    df_standard = load_test_data("test_data_standard.csv")
    df_adversarial = load_test_data("test_data_adversarial.csv")

    if df_standard is None or df_adversarial is None:
        return

    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print(f"本地评测设备: {torch.cuda.get_device_name(0)}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    models_to_test = [
        {"name": "A. 零样本基座 (mDeBERTa)", "path": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "type": "zero-shot"},
        {"name": "B. 小数据微调 (150条)", "path": os.path.join(project_root, "models", "fine_tuned_router"),
         "type": "finetuned"},
        {"name": "C. 增强数据微调 (1500条)", "path": os.path.join(project_root, "models", "fine_tuned_router_augmented"),
         "type": "finetuned"},
        {"name": "D. 外部大模型 (API)", "path": "API", "type": "api"}
    ]

    results = []

    for m in models_to_test:
        print(f"\n正在评测模型: {m['name']}")

        if m['type'] == "api":
            print("   -> 评测 [常规测试集] (API调用)...")
            std_acc, _ = evaluate_api_model(df_standard)
            print("   -> 评测 [对抗测试集] (API调用)...")
            adv_acc, avg_lat = evaluate_api_model(df_adversarial)
        else:
            print("   -> 评测 [常规测试集]...")
            std_acc, _ = evaluate_local_model(m['path'], m['type'], df_standard, device)
            print("   -> 评测 [对抗测试集]...")
            adv_acc, avg_lat = evaluate_local_model(m['path'], m['type'], df_adversarial, device)

        drop = std_acc - adv_acc
        results.append({
            "模型": m['name'],
            "常规集 Acc": f"{std_acc:.1f}%" if std_acc > 0 else "N/A",
            "对抗集 Acc": f"{adv_acc:.1f}%" if adv_acc > 0 else "N/A",
            "鲁棒性衰减": f"-{drop:.1f}%" if std_acc > 0 else "N/A",
            "单次延迟": f"{avg_lat:.1f} ms" if avg_lat > 0 else "N/A"
        })

    print("\n\n===================== 实验评测结果 =====================")
    format_row = "{:<30} | {:<12} | {:<12} | {:<12} | {:<12}"
    print(format_row.format("模型名称", "常规集 Acc", "对抗集 Acc", "鲁棒性衰减", "单次延迟"))
    print("-" * 90)
    for r in results:
        print(format_row.format(r["模型"], r["常规集 Acc"], r["对抗集 Acc"], r["鲁棒性衰减"], r["单次延迟"]))
    print("====================================================================\n")


if __name__ == "__main__":
    main()