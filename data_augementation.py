import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# 核心配置区
# ==========================================
DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "deepseek-chat"

# 扩充倍数：每条原始数据生成的新样本数
AUGMENT_MULTIPLIER = 5


def augment_text(client: OpenAI, text: str, retries: int = 3) -> list:
    """
    调用大语言模型进行文本数据增强
    """
    prompt = f"""你是一个专业的数据增强助手。
请将以下原始句子改写成 {AUGMENT_MULTIPLIER} 句意思相同，但表达方式不同的句子。

要求：
1. 核心法律诉求和事实要素保持不变。
2. 句式需多样化：长短句结合，包含不同的情绪表达或口语化表述。
3. 严格按照以下 JSON 格式返回：
{{
    "variations": ["改写句1", "改写句2", "改写句3", ...]
}}

原始句子：{text}
"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # 提高 temperature 以增加样本多样性
                response_format={"type": "json_object"},
                timeout=15
            )

            result_str = response.choices[0].message.content
            result_dict = json.loads(result_str)
            return result_dict.get("variations", [])

        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{retries}): {e}")
            time.sleep(2)  # 重试延时

    return []


def main():
    print("==================================================")
    print("法律意图数据集 - 自动化扩充引擎启动")
    print("==================================================\n")

    # 1. 设置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    input_file = os.path.join(project_root, "data", "train_data.csv")
    output_file = os.path.join(project_root, "data", "train_data_augmented.csv")

    if not os.path.exists(input_file):
        print(f"错误: 找不到原始数据文件 {input_file}")
        return

    # 2. 读取原始数据
    df_original = pd.read_csv(input_file)
    print(f"成功读取原始数据: {len(df_original)} 条")

    # 初始化结果列表，保留原始数据
    augmented_records = []
    for _, row in df_original.iterrows():
        augmented_records.append({"text": row["text"], "label": row["label"]})

    # 3. 初始化客户端
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    print(f"开始批量改写 (目标扩充 {AUGMENT_MULTIPLIER} 倍)...\n")

    # 4. 执行批量扩充
    for index, row in tqdm(df_original.iterrows(), total=len(df_original), desc="扩充进度"):
        original_text = row["text"]
        label = row["label"]

        variations = augment_text(client, original_text)

        for var_text in variations:
            if var_text.strip():
                augmented_records.append({
                    "text": var_text.strip(),
                    "label": label
                })

        # 限制请求频率以避免触发 Rate Limit
        time.sleep(0.1)

    # 5. 保存结果
    df_augmented = pd.DataFrame(augmented_records)

    # 打乱数据集顺序，防止模型训练过拟合
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    df_augmented.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("\n扩充完成！")
    print(f"数据规模变化: {len(df_original)} 条 -> {len(df_augmented)} 条")
    print(f"增强后的数据集已保存至: {output_file}")


if __name__ == "__main__":
    main()