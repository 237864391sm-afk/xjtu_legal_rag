import os

# 1. API 密钥配置
DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"

# 2. 模型配置
ROUTER_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
LLM_MODEL_NAME = "deepseek-chat"

# 3. 检索参数配置 (Top-K 召回数量)
TOP_K_STATUTE = 8
TOP_K_CRIMINAL = 2
TOP_K_CIVIL = 2

# 4. 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_STATUTES_DIR = os.path.join(DATA_DIR, "raw_statutes")
RAW_CRIMINAL_DIR = os.path.join(DATA_DIR, "raw_criminal")
RAW_CIVIL_DIR = os.path.join(DATA_DIR, "raw_civil")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")

# 5. 路由系统配置
ACTIVE_ROUTER_TYPE = "CustomBERT"
ROUTER_CONFIDENCE_THRESHOLD = 0.60
ROUTER_ENTROPY_THRESHOLD = 1.00
CUSTOM_ROUTER_MODEL_DIR = os.path.join(BASE_DIR, "models", "fine_tuned_router_augmented")