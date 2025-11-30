import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # ../pythonProject
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
os.environ["HF_METRICS_CACHE"] = str(HF_CACHE_DIR / "metrics")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import Optional, List

# model/tokenizer are lazy-loaded to avoid heavy import-time work
_tokenizer = None
_model = None
_model_name = None

# 默认使用已在本地训练好的 SFT 模型目录
DEFAULT_MODEL_PATH = PROJECT_ROOT / "hf_cache" / "qwen_0_5b_sft"


def init_model(model_name: Optional[str] = None):
    """Lazy load the tokenizer and model. If already loaded with same name, no-op."""
    global _tokenizer, _model, _model_name
    if model_name is None:

        if DEFAULT_MODEL_PATH.exists():
            model_name = str(DEFAULT_MODEL_PATH)
        else:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    if _model is not None and _model_name == model_name:
        return
    print(f"Loading model {model_name}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",
    device_map="auto",)
    _model_name = model_name
    print("Model loaded.")

# -----------------------------
# 2. 构建 RAG Prompt
# -----------------------------
def build_prompt(query, docs):
    """Construct a compact, course-style RAG prompt.

    Design goals (aligned with assignment):
    - Model answers based on the provided passages.
    - Short, factual answers (like "Brawn GP", "Terry Southern", "Grant Imahara")
    """

    # 使用更多文档提供更多上下文
    MAX_DOCS = 5
    selected_docs = docs[:MAX_DOCS]

    context_parts = []
    for i, d in enumerate(selected_docs):
        text = d.get("text", "")
        # 保留更多文本内容
        if len(text) > 500:
            text = text[:500] + "..."
        context_parts.append(f"[{i+1}]: {text}")
    context = "\n\n".join(context_parts)

    # 强调简短答案
    prompt = f"""Based on the following documents, answer the question with a short factual answer (1-5 words).

{context}

Question: {query}
Short Answer:"""

    return prompt

# -----------------------------
# 3. 模型生成答案
# -----------------------------
def llm_generate(prompt):
    # Deterministic generation to reduce hallucination
    if _tokenizer is None or _model is None:
        init_model()  # load default
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True)

    # 关键：把输入挪到模型所在设备
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = _model.generate(
        **inputs,
        max_new_tokens=100,  # 减少生成长度，避免胡编乱造
        do_sample=False,
        temperature=0.0,  # 完全确定性生成
        num_beams=1,  # 改用贪婪搜索，更快且对小模型更稳定
        early_stopping=True,
        repetition_penalty=1.2  # 避免重复
    )
    full = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full


def _parse_model_output(full_text):
    """Parse model output according to the enforced format.

    Returns (answer_str or None, evidence_str or None, source_indices list)
    If the model returned the exact 'Not found in provided documents.' string, returns (None, None, []).
    """
    # 模型输出里包含了我们拼进去的 prompt + 模型真正生成的回答。
    # 高容量模型可能会多次输出 Answer/Evidence/Sources，我们需要找到
    # "最像真正答案" 的那一块，而不是简单取最后一个 Answer:。
    txt = full_text.strip()

    # 提取所有 Answer 行
    answer_candidates = list(re.finditer(r"(?m)^\s*Answer:\s*(.+)$", txt))
    if not answer_candidates:
        return None, None, []

    best_answer = None
    best_evidence = None
    best_src_indices: List[int] = []

    # 从后往前遍历候选块，这样优先考虑靠后的生成内容
    for ans_m in reversed(answer_candidates):
        ans_text = ans_m.group(1).strip()
        # 跳过仍然是 prompt 占位符的行
        if "<one short sentence" in ans_text or "<one concise factual sentence" in ans_text:
            continue

        # 在该 Answer 行之后寻找最近的 Evidence / Sources
        span_start = ans_m.end()
        tail = txt[span_start:]

        # 尝试两种Evidence格式：
        # 格式1: Evidence: "text" (Document X)  
        # 格式2: Evidence: "text"
        ev_m = re.search(r"(?mi)^\s*Evidence:\s*\"(.+?)\"\s*\(Document\s*(\d+)\)", tail, flags=re.DOTALL | re.IGNORECASE)
        if not ev_m:
            # 尝试简化格式
            ev_m = re.search(r"(?mi)^\s*Evidence:\s*\"(.+?)\"", tail, flags=re.DOTALL | re.IGNORECASE)
        
        src_m = re.search(r"(?mi)^\s*Sources:\s*\[(.*?)\]", tail, flags=re.DOTALL | re.IGNORECASE)

        evidence = ev_m.group(1).strip() if ev_m else None
        src_indices: List[int] = []
        if ev_m and ev_m.lastindex >= 2:  # 如果有Document X在Evidence行
            try:
                src_indices.append(int(ev_m.group(2)))
            except Exception:
                pass
        if src_m and not src_indices:
            ids = re.findall(r"Document\s*(\d+)", src_m.group(1))
            src_indices = [int(i) for i in ids]

        best_answer = ans_text
        best_evidence = evidence
        best_src_indices = src_indices
        break

    return best_answer, best_evidence, best_src_indices


def _soft_verify_answer_evidence(query: str, answer: str, evidence: str, docs: Optional[List[dict]] = None) -> bool:
    """Very conservative soft verification.

    We only accept if the (normalized) answer text appears as a substring
    of either the evidence sentence or the corresponding document text.
    This strongly biases the model to "copy" from the docs instead of
    inventing new facts.
    """
    if not answer or not evidence:
        return False

    a = answer.strip().lower()
    e = evidence.strip().lower()

    # Special handling for "what faith/religion" style questions:
    # if question asks about faith/religion and the answer is a single
    # word label like "baptist", accept it when that label appears in
    # the evidence or any document text.
    q = query.strip().lower() if query else ""
    if any(k in q for k in ["what faith", "which faith", "what religion", "which religion"]):
        # take last token as potential label
        tokens = [t for t in re.findall(r"[a-zA-Z0-9_]+", a)]
        if tokens:
            label = tokens[-1]
            if label and (label in e or any(label in d.get("text", "").strip().lower() for d in (docs or []))):
                return True

    # Rule 1: Answer必须在Evidence中出现(至少关键词重叠)
    answer_words = set(a.split())
    evidence_words = set(e.split())
    
    # 至少50%的answer词汇在evidence中
    if len(answer_words) > 0:
        overlap = len(answer_words & evidence_words) / len(answer_words)
        if overlap < 0.5:
            return False
    
    # Rule 2: Evidence必须在某个文档中出现(至少80%匹配)
    if docs is not None:
        for d in docs:
            doc_text = d.get("text", "").strip().lower()
            # Evidence的主要内容必须在文档中
            if len(e) > 20:  # 如果evidence足够长
                # 检查evidence的前80%是否在文档中
                e_prefix = e[:int(len(e)*0.8)]
                if e_prefix in doc_text:
                    return True
            elif e in doc_text:  # 短evidence必须完全匹配
                return True

    return False

# -----------------------------
# 4. RAG 生成接口
# -----------------------------
def generate_answer(query, docs):
    prompt = build_prompt(query, docs)
    full = llm_generate(prompt)

    # 模型输出包含prompt + 生成的答案
    # 找到 "Short Answer:" 后面的内容
    answer = None
    
    # 找最后一个 Short Answer: 后面的内容
    if "Short Answer:" in full:
        parts = full.rsplit("Short Answer:", 1)
        if len(parts) > 1:
            answer = parts[1].strip()
            # 只取第一行
            answer = answer.split('\n')[0].strip()
            # 清理一些常见的格式问题
            answer = re.sub(r'^["\']|["\']$', '', answer)  # 移除引号
            answer = answer.strip()
    
    # 如果还是没有答案，尝试找 Answer: 或 Question: 之后
    if not answer:
        for marker in ["Answer:", "Question:"]:
            if marker in full:
                parts = full.rsplit(marker, 1)
                if len(parts) > 1:
                    after = parts[1]
                    lines = after.split('\n')
                    for line in lines[1:] if marker == "Question:" else lines:
                        line = line.strip()
                        if line and not line.startswith('[') and len(line) < 200:
                            answer = line
                            break
                if answer:
                    break
    
    # 如果还是没有答案，返回空字符串
    if not answer:
        answer = ""
    
    intermediate = {
        "top_k_used": len(docs),
        "parsed_answer": answer,
    }

    return answer, intermediate
