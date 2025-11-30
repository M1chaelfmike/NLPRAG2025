from retrieval.bm25 import bm25_retrieve
from generator.rag_llm import generate_answer
from typing import List, Optional, Dict  # 确保已有或添加这行
from generator.multi_turn import ConversationTurn  # 确保已导入 ConversationTurn
# Feature A: Multi-Turn 对话记忆 (全局实例)
_conversation_memory = None


def get_conversation_memory():
    """获取或创建对话记忆实例"""
    global _conversation_memory
    if _conversation_memory is None:
        from generator.multi_turn import ConversationMemory
        _conversation_memory = ConversationMemory(max_turns=5)
    return _conversation_memory


def clear_conversation():
    """清空对话记忆"""
    global _conversation_memory
    if _conversation_memory is not None:
        _conversation_memory.clear()


def _get_retrieval_func(method: str):
    """返回指定方法的检索函数"""
    def retrieve(query, method=method, topk=10):
        if method == "static":
            try:
                from retrieval.static_embed import embed_retrieve
                return embed_retrieve(query, topk=topk)
            except Exception:
                return bm25_retrieve(query, topk=topk)
        elif method == "hybrid":
            try:
                from retrieval.hybrid import hybrid_bm25_instruction_retrieve
                return hybrid_bm25_instruction_retrieve(query, topk=topk, alpha=0.5, mode="score")
            except Exception:
                return bm25_retrieve(query, topk=topk)
        elif method == "dense":
            try:
                from retrieval.dense import dense_retrieve
                return dense_retrieve(query, topk=topk)
            except Exception:
                return bm25_retrieve(query, topk=topk)
        elif method == "idense":
            try:
                from retrieval.instruction_dense import instruction_dense_retrieve
                return instruction_dense_retrieve(query, topk=topk)
            except Exception:
                return bm25_retrieve(query, topk=topk)
        elif method == "multivector":
            try:
                from retrieval.multivector import multivector_retrieve
                return multivector_retrieve(query, topk=topk)
            except Exception:
                return bm25_retrieve(query, topk=topk)
        else:
            return bm25_retrieve(query, topk=topk)
    return retrieve


def rag_pipeline(query, method: str = "bm25", mode: str = "basic"):
    """
    Run RAG pipeline.
    
    Args:
        query: 用户问题
        method: 检索方法 - 'bm25', 'static', 'dense', 'idense', 'hybrid', 'multivector'
        mode: 运行模式
            - 'basic': 基本单轮RAG (默认)
            - 'multi_turn': Feature A - 多轮对话，自动处理指代消解
            - 'agentic': Feature B - 智能工作流，包含问题分解、自检验证
    
    Returns:
        包含答案和中间步骤的字典
    """
    # Feature A: Multi-Turn Search
    if mode == "multi_turn":
        from generator.multi_turn import multi_turn_rag
        memory = get_conversation_memory()
        retrieval_func = lambda q, m: _get_retrieval_func(m)(q, m, topk=10)
        return multi_turn_rag(
            question=query,
            memory=memory,
            retrieval_func=retrieval_func,
            generation_func=generate_answer,
            method=method
        )
    
    # Feature B: Agentic Workflow
    if mode == "agentic":
        from generator.agentic import agentic_rag
        retrieval_func = _get_retrieval_func(method)
        return agentic_rag(query, retrieval_func, method)
    
    # Basic Single-Turn RAG (默认)
    # 1. 检索
    if method == "static":
        try:
            from retrieval.static_embed import embed_retrieve
        except Exception as e:
            # fallback to bm25 if static module not available
            print(f"Static retriever not available: {e}. Falling back to BM25.")
            docs = bm25_retrieve(query)
        else:
            docs = embed_retrieve(query, topk=10)

    elif method == "hybrid":
        try:
            from retrieval.hybrid import hybrid_bm25_instruction_retrieve
        except Exception as e:
            print(f"Hybrid retriever not available: {e}. Falling back to BM25.")
            docs = bm25_retrieve(query)
        else:
            docs = hybrid_bm25_instruction_retrieve(query, topk=10, alpha=0.5, mode="score")

    elif method == "dense":
        try:
            from retrieval.dense import dense_retrieve
        except Exception as e:
            print(f"Dense retriever not available: {e}. Falling back to BM25.")
            docs = bm25_retrieve(query)
        else:
            docs = dense_retrieve(query, topk=10)

    elif method == "idense":
        try:
            from retrieval.instruction_dense import instruction_dense_retrieve
        except Exception as e:
            print(f"Instruction-dense retriever not available: {e}. Falling back to BM25.")
            docs = bm25_retrieve(query)
        else:
            docs = instruction_dense_retrieve(query, topk=10)

    elif method == "multivector":
        try:
            from retrieval.multivector import multivector_retrieve
        except Exception as e:
            print(f"Multi-vector retriever not available: {e}. Falling back to BM25.")
            docs = bm25_retrieve(query)
        else:
            docs = multivector_retrieve(query, topk=10)

    else:
        docs = bm25_retrieve(query)

    # 2. 生成答案
    answer, intermediate = generate_answer(query, docs)

    return {
        "query": query,
        "retrieved_docs": docs,
        "answer": answer,
        "final_answer": answer,  # eval_rag.py expects this field
        "intermediate_steps": intermediate
    }

# 在main.py中添加
from generator.multi_turn import ConversationMemory, ConversationTurn

# 全局对话历史存储
conversation_memory = ConversationMemory()

def get_conversation_history() -> List[ConversationTurn]:
    """获取当前对话历史"""
    return conversation_memory.turns

def clear_conversation():
    """清空对话历史（已有函数，补充实现）"""
    global conversation_memory
    conversation_memory = ConversationMemory()  # 重新初始化