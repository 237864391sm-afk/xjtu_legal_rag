import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import evaluate
import numpy as np
from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


def main():
    print("开始训练法律意图路由器 (BERT Fine-tuning)...")

    # 1. 路径与基础配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    model_name = os.path.join(project_root, "models", "base_model")
    output_dir = os.path.join(project_root, "models", "fine_tuned_router_augmented")
    os.makedirs(output_dir, exist_ok=True)

    # 标签映射
    id2label = {0: "刑事犯罪案件", 1: "民商事纠纷", 2: "行政诉讼与国家赔偿"}
    label2id = {"刑事犯罪案件": 0, "民商事纠纷": 1, "行政诉讼与国家赔偿": 2}

    # 2. 加载数据集
    data_file_path = os.path.join(project_root, "data", "train_data_augmented.csv")

    if not os.path.exists(data_file_path):
        print(f"找不到训练数据文件: {data_file_path}")
        return

    print(f"成功定位训练数据: {data_file_path}")
    df = pd.read_csv(data_file_path)

    print("预览数据:")
    print(df.head())

    dataset = Dataset.from_pandas(df)

    # 划分训练集 (80%) 和验证集 (20%)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # 3. 加载 Tokenizer 并处理文本
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        # 截断长度设为 128 以提升训练速度
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. 加载预训练分类模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )

    # 5. 定义评估指标 (Accuracy)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 6. 设置训练超参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",          # 每个 epoch 评估一次
        save_strategy="epoch",          # 每个 epoch 保存一次
        learning_rate=2e-5,             # 微调学习率
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,             # 训练轮数
        weight_decay=0.01,              # 正则化
        fp16=True,                      # 开启半精度加速
        load_best_model_at_end=True,    # 训练结束后自动加载最优模型
    )

    # 7. 启动训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    print("开始微调...")
    trainer.train()

    # 8. 保存模型
    print(f"训练完成，模型已保存至: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()