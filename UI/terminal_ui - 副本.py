import os
import sys
import time
from pathlib import Path
from termcolor import colored

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import rag_pipeline, clear_conversation

# lazy initialization imports will be done after user choices
from retrieval import bm25 as bm25_mod
from retrieval import static_embed as static_mod
from generator import rag_llm as rag_llm_mod

def print_title():
    print(colored("=" * 70, "cyan"))
    print(colored("        🧠  Retrieval-Augmented Generation System (RAG)        ", "yellow"))
    print(colored("=" * 70, "cyan"))

def print_section(title):
    print(colored(f"\n▶ {title}", "green", attrs=["bold"]))

def print_retrieved_docs(docs, max_display=5):
    """显示检索到的文档，限制显示数量"""
    for i, d in enumerate(docs[:max_display]):
        print(colored(f"\n--- Retrieved Doc #{i+1} ---", "blue"))
        print(colored(f"[ID] {d.get('id', 'N/A')} | Score: {d.get('score', 0):.4f}", "magenta"))
        text = d.get('text', '')[:350]
        print(text + ("..." if len(d.get('text', '')) > 350 else ""))
        print(colored("-" * 50, "blue"))
    if len(docs) > max_display:
        print(colored(f"... and {len(docs) - max_display} more documents", "blue"))

def print_intermediate(steps, mode="basic"):
    """打印中间步骤，根据模式显示不同信息"""
    print_section("Intermediate Workflow")
    
    if mode == "agentic" and "agent_steps" in steps:
        # Feature B: 显示 Agent 工作流步骤
        print(colored("🤖 Agentic Workflow Steps:", "yellow", attrs=["bold"]))
        for step in steps.get("agent_steps", []):
            print(colored(f"\n  Step {step['step']} [{step['action'].upper()}]", "cyan"))
            print(colored(f"    💭 Thought: {step['thought']}", "white"))
            print(colored(f"    📥 Input: {step['input'][:80]}...", "white"))
            print(colored(f"    👁️ Observation: {step['observation'][:80]}...", "white"))
        
        # 显示自检结果
        if "self_check" in steps:
            sc = steps["self_check"]
            verdict = "✅ PASSED" if sc.get("final_verdict") else "❌ FAILED"
            print(colored(f"\n  Self-Check: {verdict}", "green" if sc.get("final_verdict") else "red"))
            print(colored(f"    Reason: {sc.get('reason', 'N/A')}", "white"))
    
    elif mode == "multi_turn" and "query_rewriting" in steps:
        # Feature A: 显示多轮对话信息
        print(colored("🔄 Multi-Turn Conversation:", "yellow", attrs=["bold"]))
        qr = steps.get("query_rewriting", {})
        print(colored(f"  Original: {qr.get('original_question', 'N/A')}", "white"))
        print(colored(f"  Rewritten: {qr.get('rewritten_query', 'N/A')}", "cyan"))
        print(colored(f"  Has Coreference: {qr.get('has_coreference', False)}", "white"))
        print(colored(f"  Conversation Turns: {steps.get('conversation_turns', 1)}", "white"))
        if steps.get("extracted_entities"):
            print(colored(f"  Entities: {steps['extracted_entities']}", "magenta"))
    
    else:
        # Basic mode: 显示简单信息
        for k, v in steps.items():
            if k not in ["agent_steps", "query_rewriting", "self_check"]:
                # 截断过长的值
                v_str = str(v)
                if len(v_str) > 100:
                    v_str = v_str[:100] + "..."
                print(colored(f"• {k}:", "yellow"), v_str)

def print_help():
    """打印帮助信息"""
    print(colored("\n📚 Available Commands:", "cyan", attrs=["bold"]))
    print(colored("  /mode basic     - Switch to basic single-turn RAG", "white"))
    print(colored("  /mode multi     - Switch to multi-turn conversation (Feature A)", "white"))
    print(colored("  /mode agentic   - Switch to agentic workflow (Feature B)", "white"))
    print(colored("  /clear          - Clear conversation history", "white"))
    print(colored("  /help           - Show this help message", "white"))
    print(colored("  exit            - Exit the program", "white"))

def run_ui():
    os.system("cls" if os.name == "nt" else "clear")
    print_title()
    
    # 选择检索方法
    method = "bm25"
    m = input(colored(
        "Choose retriever - type 'bm25', 'static', 'dense', 'idense', 'hybrid' or 'multivector' (default bm25): ",
        "cyan")).strip().lower()
    if m in ["bm25", "static", "dense", "idense", "hybrid", "multivector"]:
        method = m

    hybrid_alpha = 0.5
    if method == "hybrid":
        a = input(colored("Enter hybrid alpha for BM25 weight (0.0-1.0, default 0.5): ", "cyan")).strip()
        try:
            if a:
                hybrid_alpha = float(a)
                hybrid_alpha = max(0.0, min(1.0, hybrid_alpha))
        except Exception:
            hybrid_alpha = 0.5

    # 选择运行模式
    print(colored("\n🎯 Available RAG Modes:", "cyan", attrs=["bold"]))
    print(colored("  1) basic     - Basic single-turn RAG (default)", "white"))
    print(colored("  2) multi     - Multi-turn conversation with coreference resolution (Feature A)", "white"))
    print(colored("  3) agentic   - Agentic workflow with query decomposition & self-check (Feature B)", "white"))
    
    mode_input = input(colored("Choose mode (1/2/3 or basic/multi/agentic, default basic): ", "cyan")).strip().lower()
    mode_map = {"1": "basic", "2": "multi_turn", "3": "agentic", 
                "basic": "basic", "multi": "multi_turn", "agentic": "agentic"}
    mode = mode_map.get(mode_input, "basic")
    
    # 选择模型
    print(colored("\nAvailable models: 1) Qwen/Qwen2.5-0.5B-Instruct (default)", "cyan"))
    model_choice = input(colored("Enter model id (or press Enter for default): ", "cyan")).strip()
    if not model_choice:
        model_choice = None

    print(colored(f"\n⚙️ Configuration:", "yellow", attrs=["bold"]))
    print(colored(f"  Retriever: {method}", "white"))
    print(colored(f"  Mode: {mode}", "white"))
    print(colored(f"  Model: {model_choice or 'default'}", "white"))
    
    print(colored(f"\nInitializing...", "yellow"))

    # initialize model (lazy load)
    try:
        rag_llm_mod.init_model(model_choice)
    except Exception as e:
        print(colored(f"Warning: model init failed: {e}", "red"))

    # initialize retrieval index only for chosen method
    try:
        if method == "bm25":
            idx_path = str(bm25_mod.HF_CACHE_DIR / "bm25_idx.pkl") if hasattr(bm25_mod, "HF_CACHE_DIR") else None
            bm25_mod.init(index_path=idx_path)
        elif method == "static":
            static_mod.ensure_index()
            static_mod.load_index()
        elif method == "dense":
            try:
                from retrieval import dense as dense_mod
            except Exception as e:
                print(colored(f"Dense module not available: {e}", "red"))
                raise
            try:
                dense_mod.ensure_index()
                dense_mod.load_index()
            except Exception as e:
                print(colored(f"Warning: dense retriever init failed: {e}", "red"))
        elif method == "idense":
            try:
                from retrieval import instruction_dense as idense_mod
            except Exception as e:
                print(colored(f"Instruction-dense module not available: {e}", "red"))
                raise
            try:
                idense_mod.ensure_index()
                idense_mod.load_index()
            except Exception as e:
                print(colored(f"Warning: instruction-dense retriever init failed: {e}", "red"))
        elif method == "hybrid":
            idx_path = str(bm25_mod.HF_CACHE_DIR / "bm25_idx.pkl") if hasattr(bm25_mod, "HF_CACHE_DIR") else None
            bm25_mod.init(index_path=idx_path)
            try:
                static_mod.ensure_index()
                static_mod.load_index()
            except Exception as e:
                print(colored(f"Warning: static retriever init failed for hybrid: {e}", "red"))
        elif method == "multivector":
            try:
                from retrieval import dense as dense_mod
            except Exception as e:
                print(colored(f"Dense module not available for multi-vector: {e}", "red"))
                raise
            try:
                dense_mod.ensure_index()
                dense_mod.load_index()
            except Exception as e:
                print(colored(f"Warning: dense retriever init failed for multi-vector: {e}", "red"))

    except Exception as e:
        print(colored(f"Warning during initialization: {e}\nYou can still continue, but the selected retriever may fall back to another." , "red"))

    print(colored("✅ Initialization complete.", "green"))
    print_help()
    print(colored("\nYou may now enter questions.", "green"))

    while True:
        q = input(colored("\n> Enter your question (or 'exit', '/help'): ", "cyan"))
        
        # 处理命令
        if q.lower() == "exit":
            print(colored("Goodbye! 👋", "yellow"))
            break
        
        if q.lower() == "/help":
            print_help()
            continue
        
        if q.lower() == "/clear":
            clear_conversation()
            print(colored("🗑️ Conversation history cleared.", "green"))
            continue
        
        if q.lower().startswith("/mode "):
            new_mode = q.split(" ", 1)[1].strip().lower()
            mode_map = {"basic": "basic", "multi": "multi_turn", "agentic": "agentic"}
            if new_mode in mode_map:
                mode = mode_map[new_mode]
                print(colored(f"🔄 Mode switched to: {mode}", "green"))
                if new_mode == "multi":
                    clear_conversation()
                    print(colored("  (Conversation history cleared for new multi-turn session)", "white"))
            else:
                print(colored(f"❌ Unknown mode: {new_mode}. Use basic/multi/agentic", "red"))
            continue
        
        if q.startswith("/"):
            print(colored(f"❌ Unknown command: {q}. Type /help for available commands.", "red"))
            continue

        print(colored("\n⏳ Processing...", "yellow"))
        time.sleep(0.3)

        try:
            # 使用统一的 rag_pipeline 接口
            if method == "hybrid":
                # 对于hybrid，需要特殊处理alpha参数
                from retrieval.hybrid import hybrid_retrieve
                docs = hybrid_retrieve(q, topk=10, alpha=hybrid_alpha, mode="score")
                from generator.rag_llm import generate_answer
                answer, intermediate = generate_answer(q, docs)
                result = {
                    "query": q,
                    "retrieved_docs": docs,
                    "answer": answer,
                    "final_answer": answer,
                    "intermediate_steps": intermediate,
                }
            else:
                result = rag_pipeline(q, method=method, mode=mode)
            
            # 显示结果
            print_section("🎯 Final Answer")
            print(colored(result.get("answer", "No answer"), "yellow", attrs=["bold"]))
            
            # 如果是多轮对话模式，显示重写后的查询
            if mode == "multi_turn" and result.get("rewritten_query") != q:
                print(colored(f"\n  (Query rewritten to: {result.get('rewritten_query', q)})", "cyan"))

            print_section("📚 Retrieved Documents")
            print_retrieved_docs(result.get("retrieved_docs", []))

            print_intermediate(result.get("intermediate_steps", {}), mode=mode)

            print(colored("\n" + "=" * 70, "cyan"))
            
        except Exception as e:
            print(colored(f"❌ Error: {e}", "red"))
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_ui()
