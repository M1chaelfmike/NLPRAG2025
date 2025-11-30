"""
Feature A: Multi-Turn Search
支持多轮对话，维护对话历史，自动将后续问题重写为独立查询。
增强版：区分人物、地点、事物，智能指代消解。

Example:
Q1: "Where was Barack Obama born?" -> A1: Hawaii, USA.
Q2: "What about his wife, where was she born?" -> 重写为 "Where was Michelle Obama born?" -> A2: Chicago, Illinois, USA.
"""

import os
import pathlib
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))

from transformers import AutoModelForCausalLM, AutoTokenizer

# Lazy load model
_tokenizer = None
_model = None

# 常见人名标识词（用于判断是否为人名）
PERSON_INDICATORS = {
    "mr", "mrs", "ms", "dr", "professor", "president", "king", "queen",
    "prince", "princess", "director", "actor", "actress", "singer", "author",
    "ceo", "founder", "coach", "player", "captain", "general", "senator",
    "born", "died", "married", "wife", "husband", "son", "daughter"
}

# 常见地点标识词
PLACE_INDICATORS = {
    "city", "country", "state", "province", "town", "village", "island",
    "river", "lake", "mountain", "ocean", "sea", "continent", "region",
    "university", "college", "school", "hospital", "airport", "station",
    "located", "capital", "population"
}


def _ensure_model():
    global _tokenizer, _model
    if _model is not None:
        return
    
    # 尝试使用本地SFT模型，否则使用原始模型
    sft_path = HF_CACHE_DIR / "qwen_0_5b_sft"
    model_name = str(sft_path) if sft_path.exists() else "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"[Multi-Turn] Loading model: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    print("[Multi-Turn] Model loaded.")


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    question: str  # 原始问题
    rewritten_query: str  # 重写后的独立查询
    answer: str  # 答案
    entities: List[str] = field(default_factory=list)  # 提取的实体


class ConversationMemory:
    """对话记忆管理器 - 增强版实体追踪"""
    
    def __init__(self, max_turns: int = 5):
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns
        
        # 分类实体追踪
        self.last_person: Optional[str] = None      # 最近提到的人
        self.last_place: Optional[str] = None       # 最近提到的地点
        self.last_thing: Optional[str] = None       # 最近提到的事物/其他
        self.all_entities: List[str] = []           # 所有实体（按顺序）
        
        # 关系映射 (例如 "wife" -> "Michelle Obama")
        self.relationships: Dict[str, str] = {}
        
        # 主题追踪
        self.current_topic: Optional[str] = None
    
    def _classify_entity(self, entity: str, context: str = "") -> str:
        """判断实体类型: person, place, thing"""
        entity_lower = entity.lower()
        context_lower = context.lower()
        
        # 检查是否是书名/电影名等作品名（已知的）
        work_names = {
            "harry potter", "star wars", "lord of the rings", "game of thrones",
            "dark knight", "iron man", "spider man", "the matrix", "inception",
            "interstellar", "dunkirk", "batman", "avengers"
        }
        if entity_lower in work_names or any(w in entity_lower for w in ["movie", "book", "series", "film"]):
            return "thing"
        
        # 检查上下文是否表明这是作品名
        work_context_words = ["wrote", "directed", "movie", "book", "film", "series", "novel", "album"]
        for indicator in work_context_words:
            # 如果上下文说 "wrote X" 或 "directed X"，X可能是作品名
            pattern = rf"(?:wrote|directed|book|movie|film|series)\s+(?:the\s+)?{re.escape(entity)}"
            if re.search(pattern, context_lower, re.IGNORECASE):
                return "thing"
        
        # 先检查实体名称是否像地名
        common_places = {
            "hawaii", "honolulu", "chicago", "new york", "los angeles", "london",
            "paris", "tokyo", "beijing", "washington", "california", "texas",
            "florida", "illinois", "america", "china", "japan", "england", "france",
            "edinburgh", "scotland", "germany", "australia", "canada"
        }
        if entity_lower in common_places:
            return "place"
        
        # 检查地名后缀
        place_suffixes = ("land", "ia", "stan", "burg", "ville", "town", "city", 
                         "shire", "ford", "port", "ton", "polis")
        if any(entity_lower.endswith(suffix) for suffix in place_suffixes):
            return "place"
        
        # 检查上下文是否包含地点标识词
        place_context_words = ["born in", "located in", "city of", "state of", 
                              "capital of", "from", "lives in", "moved to", "in"]
        for indicator in place_context_words:
            pattern = rf"{indicator}\s*{re.escape(entity)}"
            if re.search(pattern, context_lower, re.IGNORECASE):
                return "place"
        
        # 检查是否有 J.K. 这样的缩写开头（通常是人名）
        if re.match(r'^[A-Z]\.[A-Z]?\.?\s*[A-Z]', entity):
            return "person"
        
        # 两个大写单词开头通常是人名 (Barack Obama, Michelle Obama)
        words = entity.split()
        if len(words) == 2 and all(w and w[0].isupper() for w in words):
            # 检查是否在已知作品名列表中
            if entity_lower not in work_names:
                return "person"
        
        # 单个词默认为地点或事物
        if len(words) == 1:
            return "place"
        
        return "thing"
    
    def add_turn(self, turn: ConversationTurn):
        """添加一轮对话"""
        self.turns.append(turn)
        
        # 合并问题和答案作为上下文
        context = f"{turn.question} {turn.answer}"
        
        # 从问题中提取主题实体（通常是问题开始提到的实体）
        question_entities = extract_entities(turn.question)
        topic_entity = question_entities[0] if question_entities else None
        
        # 分类并存储实体
        for entity in turn.entities:
            entity_type = self._classify_entity(entity, context)
            
            if entity_type == "person":
                self.last_person = entity
            elif entity_type == "place":
                self.last_place = entity
            else:
                self.last_thing = entity
            
            # 添加到有序列表（避免重复）
            if entity not in self.all_entities:
                self.all_entities.insert(0, entity)
            
            # 限制列表大小
            self.all_entities = self.all_entities[:15]
        
        # 如果问题有明确的主题实体，优先使用它作为 last_person
        if topic_entity:
            topic_type = self._classify_entity(topic_entity, context)
            if topic_type == "person":
                self.last_person = topic_entity
        
        # 提取关系（如 "his wife Michelle"）
        self._extract_relationships(context)
        
        # 更新当前主题
        if turn.entities:
            self.current_topic = turn.entities[0]
        
        # 保持最近的N轮对话
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
    
    def _extract_relationships(self, text: str):
        """从文本中提取关系词"""
        # 模式: "his/her [关系词] [实体]"
        patterns = [
            (r"(?:his|her)\s+(wife|husband|brother|sister|father|mother|son|daughter|friend)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", 1, 2),
            (r"(?:its|their)\s+(capital|president|founder|ceo|director|leader)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", 1, 2),
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:his|her)\s+(wife|husband)", 2, 1),
        ]
        
        for pattern, rel_group, entity_group in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relation_key = match.group(rel_group).lower()
                entity_value = match.group(entity_group)
                self.relationships[relation_key] = entity_value
    
    def resolve_reference(self, reference: str) -> Optional[str]:
        """解析指代词，返回对应的实体"""
        ref_lower = reference.lower().strip()
        
        # 男性人称代词 -> 人
        if ref_lower in ("he", "him", "his"):
            return self.last_person
        
        # 女性人称代词 -> 人
        if ref_lower in ("she", "her", "hers"):
            return self.last_person
        
        # 事物代词 -> 优先返回事物，其次地点
        if ref_lower in ("it", "its"):
            return self.last_thing or self.last_place or (self.all_entities[0] if self.all_entities else None)
        
        # 复数代词 -> 最近实体
        if ref_lower in ("they", "them", "their"):
            return self.all_entities[0] if self.all_entities else None
        
        # 地点指示
        if ref_lower in ("there", "here"):
            return self.last_place
        
        # 关系词
        if ref_lower in self.relationships:
            return self.relationships[ref_lower]
        
        # this/that 指向当前主题
        if ref_lower in ("that", "this"):
            return self.current_topic
        
        return None
    
    def get_context_summary(self) -> str:
        """获取对话上下文摘要"""
        if not self.turns:
            return ""
        
        summary_parts = []
        for i, turn in enumerate(self.turns[-3:], 1):  # 只用最近3轮
            summary_parts.append(f"Q{i}: {turn.question}")
            summary_parts.append(f"A{i}: {turn.answer}")
        
        # 添加实体上下文（帮助 LLM 理解指代）
        entity_hints = []
        if self.last_person:
            entity_hints.append(f"Person mentioned: {self.last_person}")
        if self.last_place:
            entity_hints.append(f"Place mentioned: {self.last_place}")
        if self.relationships:
            for rel, entity in list(self.relationships.items())[:3]:
                entity_hints.append(f"Known: {rel} = {entity}")
        
        if entity_hints:
            summary_parts.append("\n[Context]")
            summary_parts.extend(entity_hints)
        
        return "\n".join(summary_parts)
    
    def clear(self):
        """清空对话记忆"""
        self.turns.clear()
        self.last_person = None
        self.last_place = None
        self.last_thing = None
        self.all_entities.clear()
        self.relationships.clear()
        self.current_topic = None


def extract_entities(text: str) -> List[str]:
    """从文本中提取实体（简单启发式方法）"""
    entities = []
    
    # 模式1: 标准大写开头的词组（如 "Barack Obama", "Hawaii"）
    pattern1 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    
    # 模式2: 带缩写的名字（如 "J.K. Rowling", "J.R.R. Tolkien"）
    pattern2 = r'\b([A-Z]\.(?:[A-Z]\.)*\s*[A-Z][a-z]+)\b'
    
    matches = re.findall(pattern1, text)
    matches.extend(re.findall(pattern2, text))
    
    # 过滤掉常见的非实体词
    stop_words = {
        # 疑问词和句子开头词
        "The", "This", "That", "What", "Where", "When", "Who", "How", "Why",
        "Yes", "No", "Not", "Answer", "Evidence", "Sources", "Document",
        "Question", "Query", "Context", "Person", "Place", "Known",
        # 动词（句子开头）
        "Tell", "Show", "Give", "Find", "Search", "Get", "List", "Explain",
        "Describe", "Define", "Compare", "Calculate",
        # 代词
        "He", "She", "It", "They", "His", "Her", "Their", "Its",
        # 其他常见词
        "Also", "And", "But", "However", "Because", "Since", "After", "Before"
    }
    
    # 书籍/电影名称等非人名实体（两个词但不是人名）
    non_person_names = {
        "harry potter", "star wars", "lord rings", "game thrones",
        "dark knight", "iron man", "spider man", "the matrix"
    }
    
    seen = set()
    for match in matches:
        match_lower = match.lower()
        if match not in stop_words and len(match) > 2 and match not in seen:
            # 检查是否是已知的非人名
            if match_lower not in non_person_names:
                entities.append(match)
                seen.add(match)
    
    return entities


def _rule_based_rewrite(question: str, memory: ConversationMemory) -> Tuple[str, Dict[str, str]]:
    """
    基于规则的指代消解（在LLM之前先尝试规则）
    
    返回: (重写后的问题, 替换的映射)
    """
    resolved = {}
    rewritten = question
    
    # 定义替换模式（按优先级排序）
    pronoun_patterns = [
        # 所有格代词 + 关系词: "his wife", "her brother"
        (r'\b(his|her)\s+(wife|husband|brother|sister|father|mother|son|daughter|child|children)\b', 
         lambda m: _resolve_possessive_relation(m, memory, resolved)),
        
        # 所有格 + 一般名词: "his name", "her age"
        (r'\b(his|her)\s+(\w+)\b',
         lambda m: _resolve_possessive(m, memory, resolved)),
        
        # 人称代词主格 + 动词: "he was born", "she directed"
        (r'\b(he|she)\s+(\w+)',
         lambda m: _resolve_person_pronoun(m, memory, resolved)),
        
        # 人称代词宾格: "about him", "for her"
        (r'\b(about|for|with|to)\s+(him|her)\b',
         lambda m: _resolve_object_pronoun(m, memory, resolved)),
        
        # 独立人称代词 (句末或问号前): "how old is he?"
        (r'\b(he|she)\s*[?]',
         lambda m: _resolve_standalone_pronoun(m, memory, resolved)),
        
        # 独立人称代词 (句中): "is he", "was she"
        (r'\b(is|was|did|does|has|had|will|would|can|could)\s+(he|she)\b',
         lambda m: _resolve_verb_pronoun(m, memory, resolved)),
        
        # 地点指示: "there", "that place"
        (r'\b(there)\b',
         lambda m: _resolve_place(m, memory, resolved)),
        
        # "it" 代词
        (r'\bit\b',
         lambda m: _resolve_it(m, memory, resolved)),
    ]
    
    for pattern, resolver in pronoun_patterns:
        rewritten = re.sub(pattern, resolver, rewritten, flags=re.IGNORECASE)
    
    return rewritten, resolved


def _resolve_possessive_relation(match, memory: ConversationMemory, resolved: dict) -> str:
    """解析 'his wife', 'her brother' 等关系表达"""
    possessive = match.group(1).lower()
    relation = match.group(2).lower()
    
    # 先检查是否有已知关系
    if relation in memory.relationships:
        entity = memory.relationships[relation]
        resolved[match.group(0)] = entity
        return entity
    
    # 否则用 "X's [relation]" 格式
    if memory.last_person:
        replacement = f"{memory.last_person}'s {relation}"
        resolved[match.group(0)] = replacement
        return replacement
    
    return match.group(0)


def _resolve_person_pronoun(match, memory: ConversationMemory, resolved: dict) -> str:
    """解析 'he was', 'she directed' 等"""
    pronoun = match.group(1)
    verb = match.group(2)
    
    if memory.last_person:
        resolved[pronoun] = memory.last_person
        return f"{memory.last_person} {verb}"
    
    return match.group(0)


def _resolve_standalone_pronoun(match, memory: ConversationMemory, resolved: dict) -> str:
    """解析句末的独立代词 'he?', 'she?'"""
    pronoun = match.group(1)
    
    if memory.last_person:
        resolved[pronoun] = memory.last_person
        return f"{memory.last_person}?"
    
    return match.group(0)


def _resolve_verb_pronoun(match, memory: ConversationMemory, resolved: dict) -> str:
    """解析 'is he', 'was she' 等"""
    verb = match.group(1)
    pronoun = match.group(2)
    
    if memory.last_person:
        resolved[pronoun] = memory.last_person
        return f"{verb} {memory.last_person}"
    
    return match.group(0)


def _resolve_object_pronoun(match, memory: ConversationMemory, resolved: dict) -> str:
    """解析 'about him', 'for her' 等"""
    prep = match.group(1)
    pronoun = match.group(2)
    
    if memory.last_person:
        resolved[pronoun] = memory.last_person
        return f"{prep} {memory.last_person}"
    
    return match.group(0)


def _resolve_possessive(match, memory: ConversationMemory, resolved: dict) -> str:
    """解析 'his name', 'her age' 等"""
    possessive = match.group(1)
    noun = match.group(2)
    
    if memory.last_person:
        resolved[possessive] = memory.last_person
        return f"{memory.last_person}'s {noun}"
    
    return match.group(0)


def _resolve_place(match, memory: ConversationMemory, resolved: dict) -> str:
    """解析 'there' 等地点指示"""
    if memory.last_place:
        resolved[match.group(0)] = memory.last_place
        # 检查上下文，智能选择介词
        return f"in {memory.last_place}"
    return match.group(0)


def _resolve_it(match, memory: ConversationMemory, resolved: dict) -> str:
    """解析 'it' 代词"""
    # 优先返回事物，其次地点，最后任何实体
    entity = memory.last_thing or memory.last_place or (memory.all_entities[0] if memory.all_entities else None)
    if entity:
        resolved["it"] = entity
        return entity
    return match.group(0)


def rewrite_query(
    current_question: str,
    memory: ConversationMemory
) -> Tuple[str, Dict]:
    """
    将后续问题重写为独立的查询（增强版）
    
    策略:
    1. 先用规则进行简单替换
    2. 如果规则不够，再用LLM
    
    返回: (重写后的查询, 中间步骤信息)
    """
    _ensure_model()
    
    intermediate = {
        "original_question": current_question,
        "has_coreference": False,
        "resolved_entities": {},
        "method": "none"
    }
    
    # 如果没有历史对话，直接返回原问题
    if not memory.turns:
        intermediate["rewritten_query"] = current_question
        return current_question, intermediate
    
    # 检测是否有指代词需要解析
    coreference_patterns = [
        r'\b(he|she|it|they|him|her|them)\b',
        r'\b(his|her|their|its)\b',
        r'\b(this|that|these|those)\b',
        r'\b(the same|the other)\b',
        r'\bthere\b',
    ]
    
    has_coreference = any(
        re.search(pattern, current_question, re.IGNORECASE)
        for pattern in coreference_patterns
    )
    
    # 检查省略主语的问题
    if current_question.lower().startswith(("what about", "how about", "and ", "also ")):
        has_coreference = True
    
    intermediate["has_coreference"] = has_coreference
    
    if not has_coreference:
        intermediate["rewritten_query"] = current_question
        return current_question, intermediate
    
    # Step 1: 尝试基于规则的重写
    rule_rewritten, resolved = _rule_based_rewrite(current_question, memory)
    intermediate["resolved_entities"] = resolved
    
    # 如果规则成功替换了内容，直接返回
    if resolved and rule_rewritten != current_question:
        intermediate["method"] = "rule_based"
        intermediate["rewritten_query"] = rule_rewritten
        return rule_rewritten, intermediate
    
    # Step 2: 规则不够，使用LLM重写
    intermediate["method"] = "llm"
    context = memory.get_context_summary()
    
    rewrite_prompt = f"""Given the conversation history below, rewrite the follow-up question into a standalone question that can be understood without the conversation context.

Conversation History:
{context}

Follow-up Question: {current_question}

Rewrite the follow-up question by replacing pronouns (he, she, it, his, her, etc.) with the actual entity names from the conversation. Output ONLY the rewritten question, nothing else.

Rewritten Question:"""

    inputs = _tokenizer(rewrite_prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = _model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        repetition_penalty=1.2
    )
    
    full_output = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取重写后的问题
    if "Rewritten Question:" in full_output:
        rewritten = full_output.split("Rewritten Question:")[-1].strip()
    else:
        # 尝试取最后一行
        lines = [l.strip() for l in full_output.strip().split('\n') if l.strip()]
        rewritten = lines[-1] if lines else current_question
    
    # 清理重写结果
    rewritten = rewritten.strip().strip('"').strip("'")
    if not rewritten or len(rewritten) < 5:
        rewritten = current_question
    
    intermediate["rewritten_query"] = rewritten
    
    return rewritten, intermediate


def multi_turn_rag(
    question: str,
    memory: ConversationMemory,
    retrieval_func,  # 检索函数
    generation_func,  # 生成函数
    method: str = "bm25"
) -> Dict:
    """
    多轮RAG流程
    
    Args:
        question: 用户当前问题
        memory: 对话记忆对象
        retrieval_func: 检索函数，接收 (query, method) 返回文档列表
        generation_func: 生成函数，接收 (query, docs) 返回 (answer, intermediate)
        method: 检索方法
    
    Returns:
        包含答案和中间步骤的字典
    """
    # Step 1: 查询重写
    rewritten_query, rewrite_info = rewrite_query(question, memory)
    
    # Step 2: 使用重写后的查询进行检索
    docs = retrieval_func(rewritten_query, method)
    
    # Step 3: 生成答案
    answer, gen_intermediate = generation_func(rewritten_query, docs)
    
    # Step 4: 提取实体并更新记忆
    entities = extract_entities(answer)
    entities.extend(extract_entities(question))
    
    turn = ConversationTurn(
        question=question,
        rewritten_query=rewritten_query,
        answer=answer,
        entities=entities
    )
    memory.add_turn(turn)
    
    return {
        "query": question,
        "rewritten_query": rewritten_query,
        "retrieved_docs": docs,
        "answer": answer,
        "final_answer": answer,
        "intermediate_steps": {
            "query_rewriting": rewrite_info,
            "generation": gen_intermediate,
            "extracted_entities": entities,
            "conversation_turns": len(memory.turns)
        }
    }
