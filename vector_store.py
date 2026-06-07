import os
import json
import time
import numpy as np

# 强制使用国内 HuggingFace 镜像源加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 检查并导入核心算法库
try:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("缺少核心依赖！请在终端运行：")
    print("pip install faiss-cpu sentence-transformers torch")
    exit()


class LegalVectorStore:
    def __init__(self, data_dir: str, model_name: str = "BAAI/bge-large-zh-v1.5"):
        """
        初始化向量数据库引擎
        """
        self.processed_dir = os.path.join(data_dir, "processed")
        self.index_dir = os.path.join(data_dir, "indexes")
        os.makedirs(self.index_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"当前运行设备: {self.device.upper()}")
        if self.device == "cuda":
            print(f"显卡型号: {torch.cuda.get_device_name(0)}")

        print(f"正在加载语义引擎: {model_name}...")

        # 显存优化配置
        model_kwargs = {"torch_dtype": torch.float16} if self.device == "cuda" else {}

        try:
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs=model_kwargs
            )
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit()

        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"引擎初始化成功。向量维度: {self.dimension}")

        # 根据模型名称生成专属缓存文件后缀
        safe_model_suffix = model_name.split("/")[-1].replace(".", "_")

        self.stores = {
            "statute": {
                "data_file": "statutes.json",
                "index_file": f"statute_{safe_model_suffix}.index"
            },
            "criminal": {
                "data_file": "criminal_cases.json",
                "index_file": f"criminal_{safe_model_suffix}.index"
            },
            "civil": {
                "data_file": "civil_cases.json",
                "index_file": f"civil_{safe_model_suffix}.index"
            }
        }

    def load_or_build_index(self, db_type: str, batch_size: int = 32):
        if db_type not in self.stores:
            return

        conf = self.stores[db_type]
        data_path = os.path.join(self.processed_dir, conf["data_file"])
        index_path = os.path.join(self.index_dir, conf["index_file"])

        if not os.path.exists(data_path):
            print(f"找不到数据文件: {data_path}")
            return

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            return

        self.stores[db_type]["data"] = data

        # 优先加载本地缓存索引
        if os.path.exists(index_path):
            print(f"发现本地缓存 [{conf['index_file']}]，正在加载...")
            try:
                chunk = np.fromfile(index_path, dtype=np.uint8)
                self.stores[db_type]["index"] = faiss.deserialize_index(chunk)
                return
            except Exception as e:
                print(f"缓存加载失败，准备重新构建: {e}")

        print(f"正在为 [{db_type}] 构建向量索引 (共 {len(data)} 条)...")
        start_time = time.time()

        texts = [item["content"] for item in data]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        embeddings = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)

        self.stores[db_type]["index"] = index

        # 序列化并保存索引
        chunk = faiss.serialize_index(index)
        chunk.tofile(index_path)

        print(f"构建完成并保存。耗时: {time.time() - start_time:.2f} 秒\n")

    def search(self, query: str, db_type: str, top_k: int = 5):
        store = self.stores.get(db_type)
        if not store or "index" not in store:
            print(f"{db_type} 索引未加载")
            return []

        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec).astype('float32')

        distances, indices = store["index"].search(query_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            doc = store["data"][idx]
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "score": float(distances[0][i])
            })
        return results