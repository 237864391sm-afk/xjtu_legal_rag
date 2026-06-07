# xjtu_legal_rag
毕业设计代码
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

本仓库是基于大型语言模型（LLM）与检索增强生成（RAG）技术构建的**智能法律问答与判决辅助系统**。针对复杂法律咨询中常见的“刑民交叉”边界模糊问题与大模型法条幻觉问题，本系统提出并实现了一种基于联合概率分布的双重拦截意图路由架构。

## 📖 项目背景

在真实的法律应用场景中，用户的咨询往往杂乱无章，且极易在民事纠纷与刑事犯罪之间产生“刑民交叉”（如：合同违约与合同诈骗）。传统 RAG 系统采用单向度检索，极易导致“盲人摸象”式的误判。

为了解决这一痛点，本系统引入了：
1. **CustomBERT 智能路由网关**：通过微调的轻量化基座模型，一次前向传播同步计算分类置信度（$C_{max}$）与离散信息熵（$H$）。
2. **双重阈值兜底机制**：当意图置信度低于安全阈值或信息熵过高时，系统主动触发“全域无损兜底检索”，从物理层面掐断漏判源头。
3. **结构化去噪生成**：引入 LLM 证据适用性反思机制，输出包含“初步结论、法律依据、法理与事实分析、风险提示、行动建议”的五维标准法务报告。

## ✨ 核心特性

- **多模态路由网关**：支持 Zero-Shot、LLM API 与 CustomBERT 本地微调模型的热切换。
- **高性能向量检索引擎**：底层封装 `FAISS` 与 `SentenceTransformer`，支持亿级法律文本（成文法条、LeCaRDv2 刑案、C3RD 民案）的毫秒级召回与本地序列化缓存。
- **全链路自动化评测矩阵**：内置基于 LLM-as-a-Judge 的端到端防幻觉（SHR/QC）评测脚本，以及检索层 Recall/NDCG 指标计算。
- **开箱即用的可视化 UI**：基于 Streamlit 构建的前端交互界面，支持算法链路（路由拦截、召回得分、去噪日志）的实时透视。

## 📂 核心项目结构

```text
legal_rag_project/
├── data/                               # 本地知识库与数据集底座
│   ├── raw_statutes/                   # 原始法条 Word 文档
│   ├── raw_criminal/                   # LeCaRDv2 原始刑事案卷
│   ├── raw_civil/                      # C3RD 原始民商事案卷
│   ├── processed/                      # ETL 清洗后的结构化 JSON
│   └── indexes/                        # FAISS 本地序列化缓存 (.index)
├── models/                             # 模型权重存储区
│   └── fine_tuned_router_augmented/    # CustomBERT 意图路由微调权重
└── src/                                # 系统核心源码
    ├── config.py                       # 全局超参数与路径配置中心
    ├── data_processor.py               # 法律文书 ETL 清洗引擎
    ├── data_augmentation.py            # 基于 LLM 的意图数据扩充脚本
    ├── vector_store.py                 # 稠密向量检索引擎 (FAISS)
    ├── router.py                       # 智能意图路由网关
    ├── train_router.py                 # 路由基座模型微调脚本
    ├── generator.py                    # 知识增强结构化生成器
    ├── app.py                          # Streamlit 可视化交互前端
    └── evaluate_*.py                   # 检索层与生成层自动化评测套件
