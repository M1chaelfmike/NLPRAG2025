import os
import sys
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path

# 导入项目核心模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import rag_pipeline, clear_conversation
from retrieval import bm25 as bm25_mod
from retrieval import static_embed as static_mod
from generator import rag_llm as rag_llm_mod


class RAGGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG System")
        self.root.geometry("900x700")

        # 配置变量
        self.retrieval_method = tk.StringVar(value="bm25")
        self.mode = tk.StringVar(value="basic")
        self.model_choice = tk.StringVar(value="Qwen/Qwen2.5-0.5B-Instruct")
        self.hybrid_alpha = tk.DoubleVar(value=0.5)

        # 初始化UI组件
        self._create_widgets()

        # 状态变量
        self.initialized = False

    def _create_widgets(self):
        # 顶部配置区域
        config_frame = ttk.LabelFrame(self.root, text="Configuration")
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        # 检索方法选择
        ttk.Label(config_frame, text="Retrieval Method:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        methods = ["bm25", "static", "dense", "idense", "hybrid", "multivector"]
        method_combo = ttk.Combobox(config_frame, textvariable=self.retrieval_method, values=methods, state="readonly",
                                    width=10)
        method_combo.grid(row=0, column=1, padx=5, pady=5)

        # 混合检索权重 (仅hybrid模式)
        ttk.Label(config_frame, text="Hybrid Alpha (0.0-1.0):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.alpha_entry = ttk.Entry(config_frame, textvariable=self.hybrid_alpha, width=8)
        self.alpha_entry.grid(row=0, column=3, padx=5, pady=5)

        # 模式选择
        ttk.Label(config_frame, text="Mode:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        modes = ["basic", "multi_turn", "agentic"]
        mode_combo = ttk.Combobox(config_frame, textvariable=self.mode, values=modes, state="readonly", width=10)
        mode_combo.grid(row=0, column=5, padx=5, pady=5)

        # 模型选择
        ttk.Label(config_frame, text="Model:").grid(row=0, column=6, padx=5, pady=5, sticky=tk.W)
        model_entry = ttk.Entry(config_frame, textvariable=self.model_choice, width=30)
        model_entry.grid(row=0, column=7, padx=5, pady=5)

        # 初始化按钮
        init_btn = ttk.Button(config_frame, text="Initialize", command=self.initialize_system)
        init_btn.grid(row=0, column=8, padx=10, pady=5)

        # 对话显示区域
        self.conversation_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED)
        self.conversation_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 输入区域
        input_frame = ttk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.process_input)

        send_btn = ttk.Button(input_frame, text="Send", command=self.process_input)
        send_btn.pack(side=tk.RIGHT)

        # 状态栏
        self.status_var = tk.StringVar(value="Ready. Please click 'Initialize' to start.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def log(self, message, color=None):
        """在对话区域添加消息"""
        self.conversation_area.config(state=tk.NORMAL)
        if color:
            # 简单的颜色标记，实际显示效果取决于配置
            self.conversation_area.insert(tk.END, f"[{color}]{message}[/{color}]\n")
        else:
            self.conversation_area.insert(tk.END, f"{message}\n")
        self.conversation_area.see(tk.END)
        self.conversation_area.config(state=tk.DISABLED)

    def update_status(self, message):
        """更新状态栏消息"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def initialize_system(self):
        """初始化系统组件"""
        self.log("=" * 70)
        self.log("        🧠  Retrieval-Augmented Generation System (RAG)        ")
        self.log("=" * 70)

        self.update_status("Initializing system...")

        # 获取配置
        method = self.retrieval_method.get()
        mode = self.mode.get()
        model_choice = self.model_choice.get() or None
        hybrid_alpha = self.hybrid_alpha.get()
        hybrid_alpha = max(0.0, min(1.0, hybrid_alpha))

        # 显示配置信息
        self.log("\n⚙️ Configuration:")
        self.log(f"  Retriever: {method}")
        self.log(f"  Mode: {mode}")
        self.log(f"  Model: {model_choice or 'default'}")
        if method == "hybrid":
            self.log(f"  Hybrid Alpha: {hybrid_alpha}")

        # 初始化模型
        try:
            rag_llm_mod.init_model(model_choice)
            self.log("✅ Model initialized successfully")
        except Exception as e:
            self.log(f"⚠️ Warning: model init failed: {e}", "red")

        # 初始化检索器
        try:
            if method == "bm25":
                idx_path = str(bm25_mod.HF_CACHE_DIR / "bm25_idx.pkl") if hasattr(bm25_mod, "HF_CACHE_DIR") else None
                bm25_mod.init(index_path=idx_path)
            elif method == "static":
                static_mod.ensure_index()
                static_mod.load_index()
            elif method == "dense":
                from retrieval import dense as dense_mod
                dense_mod.ensure_index()
                dense_mod.load_index()
            elif method == "idense":
                from retrieval import instruction_dense as idense_mod
                idense_mod.ensure_index()
                idense_mod.load_index()
            elif method == "hybrid":
                idx_path = str(bm25_mod.HF_CACHE_DIR / "bm25_idx.pkl") if hasattr(bm25_mod, "HF_CACHE_DIR") else None
                bm25_mod.init(index_path=idx_path)
                static_mod.ensure_index()
                static_mod.load_index()
            elif method == "multivector":
                from retrieval import dense as dense_mod
                dense_mod.ensure_index()
                dense_mod.load_index()

            self.log(f"✅ {method} retriever initialized successfully")
        except Exception as e:
            self.log(f"⚠️ Warning during initialization: {e}", "red")
            self.log("You can still continue, but the selected retriever may fall back to another.")

        self.initialized = True
        self.update_status("Initialization complete. You may now enter questions.")
        self.log("\nYou may now enter questions.")
        self.log("Type '/help' for available commands.")
        self.user_input.focus()

    def print_retrieved_docs(self, docs, max_display=5):
        """显示检索到的文档"""
        self.log("\n--- Retrieved Documents ---")
        for i, d in enumerate(docs[:max_display]):
            self.log(f"\nDoc #{i + 1}")
            self.log(f"[ID] {d.get('id', 'N/A')} | Score: {d.get('score', 0):.4f}")
            text = d.get('text', '')[:350]
            self.log(text + ("..." if len(d.get('text', '')) > 350 else ""))
            self.log("-" * 50)
        if len(docs) > max_display:
            self.log(f"... and {len(docs) - max_display} more documents")

    def print_intermediate(self, steps, mode="basic"):
        """打印中间步骤"""
        self.log("\n▶ Intermediate Workflow")

        if mode == "agentic" and "agent_steps" in steps:
            self.log("🤖 Agentic Workflow Steps:")
            for step in steps.get("agent_steps", []):
                self.log(f"\n  Step {step['step']} [{step['action'].upper()}]")
                self.log(f"    💭 Thought: {step['thought']}")
                self.log(f"    📥 Input: {step['input'][:80]}...")
                self.log(f"    👁️ Observation: {step['observation'][:80]}...")

            if "self_check" in steps:
                sc = steps["self_check"]
                verdict = "✅ PASSED" if sc.get("final_verdict") else "❌ FAILED"
                self.log(f"\n  Self-Check: {verdict}")
                self.log(f"    Reason: {sc.get('reason', 'N/A')}")

        elif mode == "multi_turn" and "query_rewriting" in steps:
            self.log("🔄 Multi-Turn Conversation:")
            qr = steps.get("query_rewriting", {})
            self.log(f"  Original: {qr.get('original_question', 'N/A')}")
            self.log(f"  Rewritten: {qr.get('rewritten_query', 'N/A')}")
            self.log(f"  Has Coreference: {qr.get('has_coreference', False)}")
            self.log(f"  Conversation Turns: {steps.get('conversation_turns', 1)}")
            if steps.get("extracted_entities"):
                self.log(f"  Entities: {steps['extracted_entities']}")

        else:
            for k, v in steps.items():
                if k not in ["agent_steps", "query_rewriting", "self_check"]:
                    v_str = str(v)
                    if len(v_str) > 100:
                        v_str = v_str[:100] + "..."
                    self.log(f"• {k}: {v_str}")

    def print_help(self):
        """显示帮助信息"""
        self.log("\n📚 Available Commands:")
        self.log("  /mode basic     - Switch to basic single-turn RAG")
        self.log("  /mode multi     - Switch to multi-turn conversation (Feature A)")
        self.log("  /mode agentic   - Switch to agentic workflow (Feature B)")
        self.log("  /clear          - Clear conversation history")
        self.log("  /help           - Show this help message")
        self.log("  exit            - Exit the program")

    def process_input(self, event=None):
        user_query = self.user_input.get().strip()
        if not user_query:
            return

        # 清空输入框
        self.user_input.delete(0, tk.END)

        # 显示用户输入
        self.log(f"\n> You: {user_query}")

        if not self.initialized:
            self.log("❌ Please click 'Initialize' first to start the system.", "red")
            return

        # 处理命令（省略命令处理部分，与之前一致）
        if user_query.lower() == "/help":
            self.print_help()
            return
        # ... 其他命令处理逻辑 ...

        # 处理查询
        self.update_status("Processing query...")
        try:
            method = self.retrieval_method.get()
            mode = self.mode.get()
            hybrid_alpha = self.hybrid_alpha.get()
            hybrid_alpha = max(0.0, min(1.0, hybrid_alpha))  # 确保在0-1范围内

            # 调用 RAG pipeline，区分 hybrid 模式和其他模式
            if method == "hybrid":
                # 模仿 terminal_ui.py 中 hybrid 模式的处理逻辑
                from retrieval.hybrid import hybrid_retrieve
                from generator.rag_llm import generate_answer
                # 调用混合检索，传入 alpha 参数
                docs = hybrid_retrieve(user_query, topk=10, alpha=hybrid_alpha, mode="score")
                # 生成答案
                answer, intermediate = generate_answer(user_query, docs)
                # 构造与 rag_pipeline 一致的结果格式
                result = {
                    "query": user_query,
                    "retrieved_docs": docs,
                    "answer": answer,
                    "final_answer": answer,
                    "intermediate_steps": intermediate,
                }
            else:
                # 非 hybrid 模式直接调用 rag_pipeline，不传递 hybrid_alpha
                result = rag_pipeline(
                    user_query,
                    method=method,
                    mode=mode  # 只传递 method 和 mode，去掉 hybrid_alpha
                )

            # 显示检索到的文档
            self.print_retrieved_docs(result.get("retrieved_docs", []))

            # 显示中间步骤（多轮/Agent 模式的核心信息）
            self.print_intermediate(
                steps=result.get("intermediate_steps", {}),
                mode=mode
            )

            # 显示最终答案
            self.log("\n▶ Final Answer:")
            self.log(result.get("final_answer", "No answer generated."))

        except Exception as e:
            self.log(f"❌ Error processing query: {str(e)}", "red")
        finally:
            self.update_status("Ready")


if __name__ == "__main__":
    root = tk.Tk()
    app = RAGGUI(root)
    root.mainloop()