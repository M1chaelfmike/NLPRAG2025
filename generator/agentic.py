"""
Feature B: Agentic Workflow
实现智能工作流，包括:
1. 查询重写与规划 (Query Rewriting & Planning)
2. 自检验证 (Self-Checking)
3. 显式推理步骤 (Chain-of-Thought / ReAct style)
"""

import os
import pathlib
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))

from transformers import AutoModelForCausalLM, AutoTokenizer

# Lazy load model
_tokenizer = None
_model = None


def _ensure_model():
    global _tokenizer, _model
    if _model is not None:
        return
    
    sft_path = HF_CACHE_DIR / "qwen_0_5b_sft"
    model_name = str(sft_path) if sft_path.exists() else "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"[Agentic] Loading model: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    print("[Agentic] Model loaded.")


class AgentAction(Enum):
    """Agent 动作类型"""
    SEARCH = "search"       # 执行检索
    ANSWER = "answer"       # 生成最终答案
    VERIFY = "verify"       # 验证答案
    DECOMPOSE = "decompose" # 分解复杂问题


@dataclass
class AgentStep:
    """Agent 单步执行记录"""
    step_num: int
    action: AgentAction
    thought: str           # 思考过程
    action_input: str      # 动作输入
    observation: str       # 执行结果/观察
    

class AgenticRAG:
    """
    Agentic RAG 工作流
    
    实现 ReAct 风格的 Plan-Act-Reflect 循环:
    1. Think: 分析问题，决定下一步动作
    2. Act: 执行检索或生成
    3. Observe: 观察结果
    4. Reflect: 反思是否需要继续或已有足够信息
    """
    
    def __init__(self, retrieval_func, max_steps: int = 3):
        """
        Args:
            retrieval_func: 检索函数，接收 (query, method, topk) 返回文档列表
            max_steps: 最大执行步数，防止无限循环
        """
        self.retrieval_func = retrieval_func
        self.max_steps = max_steps
        self.steps: List[AgentStep] = []
    
    def _llm_generate(self, prompt: str, max_tokens: int = 150) -> str:
        """调用LLM生成"""
        _ensure_model()
        
        inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        device = next(_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            repetition_penalty=1.2
        )
        
        return _tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def decompose_query(self, query: str) -> Tuple[List[str], Dict]:
        """
        分解复杂查询为子问题
        
        对于多跳问题 (multi-hop)，尝试分解为多个简单问题
        """
        intermediate = {"original_query": query, "is_complex": False, "sub_queries": []}
        
        # 检测是否是复杂/多跳问题
        complex_patterns = [
            r'\b(and|also|as well as)\b.*\?',  # 多个并列问题
            r'\b(who|what|where).*\b(who|what|where)',  # 多个疑问词
            r'\b(before|after|while|when)\b.*\b(did|was|were)',  # 时间关系
            r'\b(compared to|versus|vs\.?)\b',  # 比较
        ]
        
        is_complex = any(re.search(p, query, re.IGNORECASE) for p in complex_patterns)
        
        # 如果问题较长也可能是复杂问题
        if len(query.split()) > 15:
            is_complex = True
        
        intermediate["is_complex"] = is_complex
        
        if not is_complex:
            intermediate["sub_queries"] = [query]
            return [query], intermediate
        
        # 使用LLM分解问题
        decompose_prompt = f"""Break down this complex question into simpler sub-questions that can be answered independently.

Question: {query}

If the question is already simple, just output the original question.
Output each sub-question on a new line, numbered like:
1. First sub-question
2. Second sub-question

Sub-questions:"""

        output = self._llm_generate(decompose_prompt, max_tokens=100)
        
        # 解析子问题
        sub_queries = []
        lines = output.split('\n')
        for line in lines:
            # 匹配 "1. question" 或 "- question" 格式
            match = re.match(r'^\s*(?:\d+[\.\)]\s*|-\s*)(.*\?)', line.strip())
            if match:
                sub_q = match.group(1).strip()
                if sub_q and len(sub_q) > 5:
                    sub_queries.append(sub_q)
        
        # 如果分解失败，使用原问题
        if not sub_queries:
            sub_queries = [query]
        
        intermediate["sub_queries"] = sub_queries
        return sub_queries, intermediate
    
    def self_check(self, query: str, answer: str, evidence: str, docs: List[Dict]) -> Tuple[bool, str, Dict]:
        """
        自检验证：检查答案是否有文档支持，是否有幻觉
        
        Returns:
            (is_valid, reason, intermediate_info)
        """
        intermediate = {
            "answer_checked": answer,
            "evidence_checked": evidence,
            "checks_performed": []
        }
        
        # Check 1: 答案是否为空或无效
        if not answer or answer.strip() == "Not found in provided documents.":
            intermediate["checks_performed"].append({"check": "non_empty", "passed": False})
            return True, "Answer indicates no information found", intermediate
        
        intermediate["checks_performed"].append({"check": "non_empty", "passed": True})
        
        # Check 2: 问题关键词是否在检索文档中 (判断检索是否相关)
        query_lower = query.lower()
        query_words = set(re.findall(r'\b[a-z]+\b', query_lower))
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "how", "many", "what", "who", "where", "when", "which", "do", "does", "did"}
        query_keywords = query_words - stop_words
        
        # 检查最相关的文档是否包含问题关键词
        top_doc_text = docs[0].get("text", "").lower() if docs else ""
        query_match_count = sum(1 for kw in query_keywords if kw in top_doc_text)
        query_relevance = query_match_count / max(len(query_keywords), 1)
        
        intermediate["checks_performed"].append({
            "check": "query_relevance", 
            "passed": query_relevance > 0.3,
            "relevance": query_relevance,
            "query_keywords": list(query_keywords)[:5]
        })
        
        # 如果检索结果与问题不相关，答案很可能是幻觉
        if query_relevance < 0.2:
            intermediate["final_verdict"] = False
            intermediate["reason"] = "Retrieved documents are not relevant to the question"
            return False, "Retrieved documents are not relevant to the question - answer may be hallucinated", intermediate
        
        # Check 2.5: 检查答案是否只是从文档中抓取了无关数字/事实
        # 这是一个常见的幻觉模式：模型看到文档中有数字，就用这个数字回答
        answer_lower = answer.lower()
        
        # 检测答案中的数字
        answer_numbers = re.findall(r'\b\d+\b', answer_lower)
        if answer_numbers:
            # 检查这些数字是否在问题的上下文中有意义
            # 如果答案包含数字，但问题关键词在同一个文档句子中没有出现，可能是幻觉
            number_context_valid = False
            for doc in docs[:3]:  # 只检查前3个文档
                doc_text = doc.get("text", "").lower()
                sentences = doc_text.split('.')
                for sent in sentences:
                    # 检查句子是否同时包含问题关键词和答案中的数字
                    has_query_kw = any(kw in sent for kw in query_keywords)
                    has_answer_num = any(num in sent for num in answer_numbers)
                    if has_query_kw and has_answer_num:
                        number_context_valid = True
                        break
                if number_context_valid:
                    break
            
            intermediate["checks_performed"].append({
                "check": "number_context_valid",
                "passed": number_context_valid,
                "answer_numbers": answer_numbers
            })
            
            if not number_context_valid and len(answer_numbers) > 0:
                intermediate["final_verdict"] = False
                intermediate["reason"] = "Answer contains numbers that may not be relevant to the question"
                return False, "Answer may contain irrelevant numbers from unrelated context", intermediate
        
        # Check 3: Evidence 是否在文档中存在
        evidence_found = False
        if evidence:
            evidence_lower = evidence.lower().strip()
            for doc in docs:
                doc_text = doc.get("text", "").lower()
                # 检查evidence的大部分是否在文档中
                if len(evidence_lower) > 20:
                    # 对于长evidence，检查前80%是否匹配
                    check_portion = evidence_lower[:int(len(evidence_lower) * 0.8)]
                    if check_portion in doc_text:
                        evidence_found = True
                        break
                elif evidence_lower in doc_text:
                    evidence_found = True
                    break
        
        intermediate["checks_performed"].append({"check": "evidence_in_docs", "passed": evidence_found})
        
        # Check 3: 答案关键词是否在文档中
        answer_lower = answer.lower()
        answer_words = set(answer_lower.split())
        # 去除停用词
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or"}
        answer_keywords = answer_words - stop_words
        
        keywords_found = 0
        for doc in docs:
            doc_text = doc.get("text", "").lower()
            for keyword in answer_keywords:
                if keyword in doc_text:
                    keywords_found += 1
        
        keyword_coverage = keywords_found / max(len(answer_keywords), 1)
        intermediate["checks_performed"].append({
            "check": "answer_keywords_in_docs", 
            "passed": keyword_coverage > 0.3,
            "coverage": keyword_coverage
        })
        
        # 综合判断
        is_valid = evidence_found or keyword_coverage > 0.3
        
        if not is_valid:
            reason = "Answer may contain hallucination: evidence not found in documents"
        else:
            reason = "Answer appears to be grounded in documents"
        
        intermediate["final_verdict"] = is_valid
        intermediate["reason"] = reason
        
        return is_valid, reason, intermediate
    
    def run(self, query: str, method: str = "bm25") -> Dict:
        """
        执行 Agentic RAG 工作流
        
        Returns:
            包含最终答案和所有中间步骤的字典
        """
        self.steps = []
        all_docs = []
        final_answer = None
        
        # Step 1: 问题分解
        sub_queries, decompose_info = self.decompose_query(query)
        
        self.steps.append(AgentStep(
            step_num=1,
            action=AgentAction.DECOMPOSE,
            thought=f"Analyzing query complexity. Found {len(sub_queries)} sub-queries.",
            action_input=query,
            observation=f"Sub-queries: {sub_queries}"
        ))
        
        # Step 2: 对每个子问题执行检索
        sub_answers = []
        for i, sub_q in enumerate(sub_queries):
            # 检索
            docs = self.retrieval_func(sub_q, method, topk=10)
            all_docs.extend(docs)
            
            self.steps.append(AgentStep(
                step_num=len(self.steps) + 1,
                action=AgentAction.SEARCH,
                thought=f"Searching for information about: {sub_q}",
                action_input=sub_q,
                observation=f"Retrieved {len(docs)} documents"
            ))
        
        # 去重文档 (按id)
        seen_ids = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = doc.get("id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        # 只保留前10个文档
        unique_docs = unique_docs[:10]
        
        # Step 3: 生成答案 (使用原始的 generate_answer)
        from generator.rag_llm import generate_answer
        answer, gen_intermediate = generate_answer(query, unique_docs)
        
        self.steps.append(AgentStep(
            step_num=len(self.steps) + 1,
            action=AgentAction.ANSWER,
            thought="Generating answer based on retrieved documents",
            action_input=f"Query: {query}, Docs: {len(unique_docs)}",
            observation=f"Generated answer: {answer[:100]}..."
        ))
        
        # Step 4: 自检验证
        evidence = gen_intermediate.get("parsed_evidence", "")
        is_valid, reason, verify_info = self.self_check(query, answer, evidence, unique_docs)
        
        self.steps.append(AgentStep(
            step_num=len(self.steps) + 1,
            action=AgentAction.VERIFY,
            thought="Verifying answer against retrieved documents",
            action_input=f"Answer: {answer}, Evidence: {evidence}",
            observation=f"Valid: {is_valid}, Reason: {reason}"
        ))
        
        # 如果验证失败，返回更诚实的回答
        if not is_valid:
            # 检查是否是因为检索不相关
            if "not relevant" in reason.lower():
                final_answer = "I couldn't find relevant information in the knowledge base to answer this question."
            else:
                final_answer = "I'm not confident in my answer based on the available documents."
            
            self.steps.append(AgentStep(
                step_num=len(self.steps) + 1,
                action=AgentAction.ANSWER,
                thought="Verification failed. Providing honest response.",
                action_input=f"Original answer: {answer}",
                observation=f"Replaced with: {final_answer}"
            ))
        else:
            final_answer = answer
        
        return {
            "query": query,
            "final_answer": final_answer,
            "answer": final_answer,
            "retrieved_docs": unique_docs,
            "intermediate_steps": {
                "query_decomposition": decompose_info,
                "sub_queries": sub_queries,
                "generation": gen_intermediate,
                "self_check": verify_info,
                "agent_steps": [
                    {
                        "step": s.step_num,
                        "action": s.action.value,
                        "thought": s.thought,
                        "input": s.action_input,
                        "observation": s.observation
                    }
                    for s in self.steps
                ],
                "total_steps": len(self.steps)
            }
        }


def agentic_rag(query: str, retrieval_func, method: str = "bm25") -> Dict:
    """
    便捷函数：执行 Agentic RAG
    
    Args:
        query: 用户问题
        retrieval_func: 检索函数，接收 (query, method, topk) 返回文档列表
        method: 检索方法
    
    Returns:
        包含答案和中间步骤的字典
    """
    agent = AgenticRAG(retrieval_func)
    return agent.run(query, method)
