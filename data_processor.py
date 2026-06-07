import json
import os
import glob
import re
import csv

try:
    import docx
except ImportError:
    print("缺少依赖: pip install python-docx")


class LegalDataETL:
    def __init__(self, base_data_dir: str):
        self.base_dir = base_data_dir
        self.raw_statutes_dir = os.path.join(base_data_dir, "raw_statutes")
        self.raw_criminal_dir = os.path.join(base_data_dir, "raw_criminal")
        self.raw_civil_dir = os.path.join(base_data_dir, "raw_civil")
        self.processed_dir = os.path.join(base_data_dir, "processed")

        for d in [self.raw_statutes_dir, self.raw_criminal_dir, self.raw_civil_dir, self.processed_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

    # 1. 解析法条 Word 文档
    def process_statutes(self, output_name="statutes.json"):
        print("开始解析法条 Word 文档...")
        processed_docs = []
        article_pattern = re.compile(r"^[\s ]*(第[一二三四五六七八九十百千]+条)[\s ]*(.*)")
        file_paths = glob.glob(os.path.join(self.raw_statutes_dir, "**/*.docx"), recursive=True)

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            level = "司法解释" if "interpretation" in file_path.lower() else "基本法律/法律"
            law_name = re.sub(r"_\d{8}\.docx$", "", filename).replace(".docx", "")
            try:
                doc = docx.Document(file_path)
                curr_id, curr_content = None, []

                def save():
                    if curr_id and curr_content:
                        processed_docs.append({
                            "id": f"{law_name}-{curr_id}",
                            "content": f"【来源层级：{level}】《{law_name}》{curr_id}：{chr(10).join(curr_content)}",
                            "metadata": {"type": "statute", "law_name": law_name, "level": level}
                        })

                for para in doc.paragraphs:
                    text = para.text.strip()
                    if not text: continue
                    match = article_pattern.match(text)
                    if match:
                        save()
                        curr_id, curr_content = match.group(1), [match.group(2)] if match.group(2) else []
                    elif curr_id:
                        curr_content.append(text)
                save()
            except Exception as e:
                pass
        self._save_json(processed_docs, output_name)
        return processed_docs

    # 2. 处理刑事案例 (LeCaRDv2)
    def process_lecard_cases(self, output_name="criminal_cases.json"):
        print("开始处理刑事案例...")
        cases = []
        file_paths = []
        for ext in ["**/*.json", "**/*.jsonl"]:
            file_paths.extend(glob.glob(os.path.join(self.raw_criminal_dir, ext), recursive=True))
        cand_files = [p for p in file_paths if not any(x in p.lower() for x in ["query", "label", "others"])]

        for f_path in cand_files:
            with open(f_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    items = [data] if "pid" in data else list(data.values()) if isinstance(data, dict) else data
                    for item in items:
                        if not isinstance(item, dict): continue
                        pid = str(item.get("pid", item.get("id", "")))
                        if not pid: continue
                        qw = item.get("qw", item.get("fact", ""))
                        charge = item.get("charge", [])

                        # 提取元数据
                        metadata = {"type": "criminal_case", "case_id": pid, "charge": charge}
                        exclude_keys = {"qw", "fact", "pid", "id", "charge"}
                        for k, v in item.items():
                            if k not in exclude_keys:
                                metadata[k] = v

                        cases.append({
                            "id": f"刑案-LeCaRDv2-{pid}",
                            "content": f"【刑事类案】罪名：{','.join(charge)}。原文：\n{qw}",
                            "metadata": metadata
                        })
                except:
                    continue
        self._save_json(cases, output_name)
        return cases

    # 3. 处理民事案例 (C3RD)
    def process_c3rd_cases(self, output_name="civil_cases.json"):
        print("开始处理民事案例...")
        cases = []
        file_paths = []
        for ext in ["**/*.json", "**/*.JSON"]:
            file_paths.extend(glob.glob(os.path.join(self.raw_civil_dir, ext), recursive=True))

        for f_path in file_paths:
            with open(f_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    items = []

                    if isinstance(data, dict):
                        if any(k in data for k in ["rid", "ajmc", "qw", "query", "JudgeResult", "q_i"]):
                            items = [data]
                        else:
                            items = [v for v in data.values() if isinstance(v, dict)]
                    elif isinstance(data, list):
                        items = [v for v in data if isinstance(v, dict)]

                    for item in items:
                        cid = str(item.get("rid", item.get("q_i", item.get("id", os.path.basename(f_path).split('.')[0]))))
                        title = item.get("ajmc", "民事案件")

                        qw = item.get("qw", "")
                        if not qw:
                            fact = item.get("ajms", item.get("query", ""))
                            result = item.get("ajjg", item.get("JudgeResult", ""))
                            qw = f"【案情描述】：{fact}\n【裁判结果】：{result}"

                        metadata = {
                            "type": "civil_case",
                            "case_name": title,
                            "case_id": cid
                        }

                        exclude_keys = {"qw", "ajms", "ajjg", "query", "JudgeResult", "q_i", "rid", "id", "ajmc"}
                        for k, v in item.items():
                            if k not in exclude_keys:
                                metadata[k] = v

                        cases.append({
                            "id": f"民案-C3RD-{cid}",
                            "content": f"【民事类案】案名：{title}。\n{qw}",
                            "metadata": metadata
                        })
                except Exception as e:
                    print(f"解析 {os.path.basename(f_path)} 失败: {e}")

        self._save_json(cases, output_name)
        return cases

    # 4. 处理辅助与评测数据
    def process_lecard_eval_data(self):
        print("开始处理评测与辅助数据...")
        q_paths = glob.glob(os.path.join(self.raw_criminal_dir, "**/test_query.json"), recursive=True)
        if q_paths:
            queries = []
            with open(q_paths[0], 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        queries.append({"qid": str(d['id']), "query": d['query'], "fact": d.get('fact', '')})
            self._save_json(queries, "test_queries.json")

        sw = glob.glob(os.path.join(self.raw_criminal_dir, "**/stopword.txt"), recursive=True)
        if sw:
            with open(sw[0], 'r', encoding='utf-8') as f:
                self._save_json([l.strip() for l in f if l.strip()], "stopwords.json")

        csv_files = glob.glob(os.path.join(self.raw_criminal_dir, "**/*.csv"), recursive=True)
        if csv_files:
            keywords = []
            with open(csv_files[0], 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip(): keywords.append(row[0].strip())
            self._save_json(keywords, "procedural_keywords.json")

    def _save_json(self, data, filename):
        if not data: return
        out_path = os.path.join(self.processed_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"成功保存 {len(data)} 条数据至: {filename}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_folder = os.path.join(project_root, "data")

    etl = LegalDataETL(data_folder)

    etl.process_statutes()
    etl.process_lecard_cases()
    etl.process_c3rd_cases()
    etl.process_lecard_eval_data()