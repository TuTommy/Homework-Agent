from typing import List, Dict, Any, Annotated, Sequence, Literal, TypedDict, Union
from datetime import datetime
import operator
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.tools import tool

# 配置模型
# 从当前文件夹的 .env 文件读取 API Key，不把密码写进代码。
load_dotenv(Path(__file__).with_name(".env"))

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("没有找到 DASHSCOPE_API_KEY，请检查 .env 文件。")

model = ChatOpenAI(
    model="qwen-turbo",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=0.7
)

# 长期记忆文件：程序运行后会自动生成。
LONG_MEMORY_FILE = Path(__file__).with_name("long_term_memory.json")
# 每次运行的完整辩论记录会保存到 outputs 文件夹。
OUTPUT_DIR = Path(__file__).with_name("outputs")
TOOL_LOG_FILE = OUTPUT_DIR / "tool_calls.json"

# 定义搜索工具
@tool
def search_web(query: str) -> str:
    """Search the web for information about AI in education."""
    # 简化实现，返回预设的搜索结果
    results = {
        "ai benefits": "研究表明，AI辅助教学可以提高学生成绩平均15-20%，个性化学习体验显著提升学习效果。",
        "ai concerns": "教育专家警告：过度依赖AI可能导致学生批判性思维能力下降，且存在数据隐私安全风险。",
        "teacher impact": "调查显示：40%的教师认为AI能有效减轻工作负担，但60%担心可能影响师生互动质量。",
        "student feedback": "学生反馈：78%的学生认为AI辅助工具帮助提高学习效率，但也有22%表示担心依赖性问题。"
    }
    # 返回最相关的结果，如果没有匹配则返回通用信息
    for key, value in results.items():
        if key in query.lower():
            return value
    return "目前教育领域的AI应用正在快速发展，需要平衡创新与传统教育方式的优势。"

# 创建搜索工具实例
search_tool = search_web

# 定义第二种工具：根据论点、证据和反驳数量计算辩论质量分。
@tool
def calculate_debate_score(
    argument_count: int,
    evidence_count: int,
    rebuttal_count: int
) -> str:
    """计算辩论质量分。参数分别表示论点、证据和反驳的数量。"""
    score = argument_count * 4 + evidence_count * 3 + rebuttal_count * 2
    return (
        f"辩论质量分 = {argument_count}*4 + "
        f"{evidence_count}*3 + {rebuttal_count}*2 = {score} 分。"
    )

# 创建计算工具实例，后面 Con1 会调用它。
score_tool = calculate_debate_score

def load_long_term_memory() -> str:
    """读取上一次辩论的教师总结。"""
    if not LONG_MEMORY_FILE.exists():
        return "（暂无历史辩论记录）"

    try:
        data = json.loads(LONG_MEMORY_FILE.read_text(encoding="utf-8"))
        return data.get("summary", "（历史记录中没有总结）")
    except (json.JSONDecodeError, OSError):
        return "（历史记忆文件无法读取）"


def save_long_term_memory(summary: str) -> None:
    """把本轮教师总结保存到 JSON 文件。"""
    record = {
        "saved_at": datetime.now().isoformat(),
        "summary": summary
    }
    LONG_MEMORY_FILE.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def save_debate_record(messages: List[Dict], final_summary: str) -> Path:
    """保存本次辩论的所有发言和教师总结。"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    record = {
        "saved_at": datetime.now().isoformat(),
        "messages": messages,
        "final_summary": final_summary
    }

    file_name = datetime.now().strftime("debate_%Y%m%d_%H%M%S.json")
    record_path = OUTPUT_DIR / file_name

    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return record_path

def append_tool_log(
    agent_name: str,
    tool_name: str,
    arguments: Dict,
    result: str
) -> None:
    """把一次工具调用追加保存到 JSON 文件。"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if TOOL_LOG_FILE.exists():
        logs = json.loads(TOOL_LOG_FILE.read_text(encoding="utf-8"))
    else:
        logs = []

    logs.append({
        "called_at": datetime.now().isoformat(),
        "agent": agent_name,
        "tool": tool_name,
        "arguments": arguments,
        "result": result
    })

    TOOL_LOG_FILE.write_text(
        json.dumps(logs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

# 定义状态类型
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    current_speaker: str
    round: int
    max_rounds: int
    terminated: bool

# 定义Agent的基础类
class DebateAgent:
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.memory: List[Dict] = []

    def update_memory(self, message: Dict):
        self.memory.append(message)

    def get_context(self, messages: List[Dict]) -> str:
        # 短期记忆：所有 Agent 都读取整场辩论最近 5 条消息。
        recent_messages = messages[-5:]
        memory_str = "\n".join(
            [f"{m['speaker']}: {m['content']}" for m in recent_messages]
        )
        long_memory = load_long_term_memory()
        return f"""
{self.system_prompt}

历史对话：
{memory_str}
历史辩论摘要（长期记忆）：
{long_memory}

现在轮到你发言。请根据历史对话和你的角色，继续进行讨论。
"""

# 创建辩论参与者
teacher = DebateAgent(
    "Teacher",
    "moderator",
    "你是一位经验丰富的教师，正在主持一场关于人工智能是否应该在教育领域全面应用的课堂辩论。"
    "你的职责是引导讨论、确保辩论有序进行，并在适当时候总结观点。"
)

pro_team = [
    DebateAgent(
        "Pro1",
        "supporter",
        "你是正方一号，专门从教学效率角度发言。"
        "论证 AI 如何减少批改、答疑和统计等重复工作，"
        "并说明教师可把节省的时间投入学生指导。"
    ),
    DebateAgent(
        "Pro2",
        "supporter",
        "你是正方二号，专门从个性化学习角度发言。"
        "论证 AI 如何根据学生能力、进度和错误情况提供分层练习与学习路径。"
    ),
    DebateAgent(
        "Pro3",
        "supporter",
        "你是正方三号，专门从教育创新和资源公平角度发言。"
        "论证 AI 如何帮助偏远地区获取学习资源，并讨论人机协作的使用边界。"
    )
]

# 创建反方辩手团队
con_team = []
# 添加具有ReAct能力的Con1
con_team.append(
    DebateAgent("Con1", "opponent",
        """你是反对在教育领域全面应用人工智能的一方。你具有搜索工具来获取实时信息。
        在发言前，你会先思考需要搜索什么信息，然后基于搜索结果作出回应。
        你的发言应该遵循以下格式：
        思考：让我思考需要搜索什么信息...
        搜索：[你要搜索的内容]
        发现：[搜索结果]
        回应：[基于搜索结果的论述]""")
)
# 添加其他反方辩手
con_team.append(
    DebateAgent(
        "Con2",
        "opponent",
        "你是反方二号，专门从师生关系和教育本质角度发言。"
        "说明教师的情感支持、价值引导和课堂观察为何难以被 AI 完全替代。"
    )
)

con_team.append(
    DebateAgent(
        "Con3",
        "opponent",
        "你是反方三号，专门从学习主体性、隐私和公平风险角度发言。"
        "讨论学生过度依赖、数据隐私、算法偏见和数字鸿沟，并提出治理建议。"
    )
)

# 定义节点函数
def teacher_node(state: AgentState, is_summary: bool = False) -> Dict:
    """教师节点处理函数"""
    context = teacher.get_context(state["messages"])
    response = model.invoke([HumanMessage(content=context)])

    message = {
        "speaker": teacher.name,
        "content": response.content,
        "timestamp": datetime.now().isoformat()
    }

    teacher.update_memory(message)
    state["messages"].append(message)
    state["current_speaker"] = teacher.name
    print(f"\n{teacher.name}: {response.content}")

    if is_summary:
    # 保存教师总结，作为下一次辩论可读取的长期记忆。
        save_long_term_memory(response.content)
        print(f"长期记忆已保存到：{LONG_MEMORY_FILE}")

        # 保存本轮所有 Agent 的发言，供实验报告查看。
        record_path = save_debate_record(state["messages"], response.content)
        print(f"完整辩论记录已保存到：{record_path}")

        return {"state": state, "next": None}

    # 开场教师不保存记忆。
    return {"state": state, "next": "pro1"}

def pro_node(state: AgentState, agent: DebateAgent) -> Dict:
    """正方辩手节点处理函数"""
    context = agent.get_context(state["messages"])
    response = model.invoke([HumanMessage(content=context)])

    message = {
        "speaker": agent.name,
        "content": response.content,
        "timestamp": datetime.now().isoformat()
    }

    agent.update_memory(message)
    state["messages"].append(message)
    state["current_speaker"] = agent.name

    print(f"\n{agent.name}: {response.content}")
    # 正方发言后，转给对应轮次的反方
    round_num = int(agent.name[-1])
    return {"state": state, "next": f"con{round_num}"}

def run_con1_react(context: str):
    """让 Con1 先调用工具，再根据工具结果完成反方回应。"""

    # 把两个工具交给模型，模型决定调用时使用什么参数。
    react_model = model.bind_tools([search_tool, score_tool])

    react_prompt = f"""
{context}

你是 Con1，负责从反方角度论证 AI 不应在教育领域全面应用。

请先选择并调用对当前辩论有帮助的工具：
- search_web：检索 AI 教育的风险或教师影响；
- calculate_debate_score：计算当前论证的质量分。

工具调用完成后，程序会把结果交给你。现在先进行工具调用。
"""

    first_response = react_model.invoke([HumanMessage(content=react_prompt)])
    observations = []

    # 执行模型实际请求的工具，并记录“观察结果”。
    for call in getattr(first_response, "tool_calls", []):
        tool_name = call["name"]
        tool_args = call["args"]

        if tool_name == search_tool.name:
            result = search_tool.invoke(tool_args)
        elif tool_name == score_tool.name:
            result = score_tool.invoke(tool_args)
        else:
            continue

        print(f"[Con1 工具调用] {tool_name} -> {result}")
        # 保存工具调用证据，之后可用于实验报告。
        append_tool_log("Con1", tool_name, tool_args, str(result))
        observations.append(f"{tool_name} 的结果：{result}")

    if not observations:
        observations.append("本轮模型没有调用工具，请依据已有对话谨慎回应。")

    final_prompt = f"""
{context}

Con1 的工具观察结果：
{chr(10).join(observations)}

请根据以上观察结果，用反方身份给出一段简洁、有依据的回应。
"""

    return model.invoke([HumanMessage(content=final_prompt)])

def con_node(state: AgentState, agent: DebateAgent) -> Dict:
    """反方辩手节点处理函数"""
    context = agent.get_context(state["messages"])

    # 为Con1添加ReAct流程
    # Con1 使用 ReAct：模型调用工具，程序执行工具，再生成最终回应。
    if agent.name == "Con1":
        response = run_con1_react(context)
    else:
        response = model.invoke([HumanMessage(content=context)])

    message = {
        "speaker": agent.name,
        "content": response.content,
        "timestamp": datetime.now().isoformat()
    }

    agent.update_memory(message)
    state["messages"].append(message)
    state["current_speaker"] = agent.name
    state["round"] += 1
    print(f"\n{agent.name}: {response.content}")

    # 如果还没到最大轮次，转给下一轮的正方，否则回到教师总结
    if state["round"] < state["max_rounds"]:
        return {"state": state, "next": f"pro{state['round'] + 1}"}
    else:
        return {"state": state, "next": "teacher"}

# 构建图
def build_debate_graph() -> Any:
    """构建只运行一轮、最后由教师总结的辩论流程。"""
    workflow = StateGraph(AgentState)

    # 同一个教师 Agent 在开场和总结时各使用一次，
    # 但在图中使用不同节点名称，避免流程回到开头后无限循环。
    workflow.add_node(
        "teacher_opening",
        lambda state: teacher_node(state, is_summary=False)
    )
    workflow.add_node("pro1", lambda state: pro_node(state, pro_team[0]))
    workflow.add_node("con1", lambda state: con_node(state, con_team[0]))
    workflow.add_node("pro2", lambda state: pro_node(state, pro_team[1]))
    workflow.add_node("con2", lambda state: con_node(state, con_team[1]))
    workflow.add_node("pro3", lambda state: pro_node(state, pro_team[2]))
    workflow.add_node("con3", lambda state: con_node(state, con_team[2]))
    workflow.add_node(
        "teacher_summary",
        lambda state: teacher_node(state, is_summary=True)
    )

    # 固定的单轮辩论顺序。
    workflow.add_edge("teacher_opening", "pro1")
    workflow.add_edge("pro1", "con1")
    workflow.add_edge("con1", "pro2")
    workflow.add_edge("pro2", "con2")
    workflow.add_edge("con2", "pro3")
    workflow.add_edge("pro3", "con3")
    workflow.add_edge("con3", "teacher_summary")

    # 教师总结后明确结束，而不是回到开场教师。
    workflow.add_edge("teacher_summary", END)

    workflow.set_entry_point("teacher_opening")
    return workflow.compile()

# 主函数
def main():
    print("\n=== 开始构建辩论图 ===")
    graph = build_debate_graph()
    print("图构建完成")

    # 初始化状态
    initial_state = AgentState(
        messages=[{
            "speaker": "System",
            "content": "让我们开始关于人工智能是否应该在教育领域全面应用的课堂讨论。请教师主持讨论。",
            "timestamp": datetime.now().isoformat()
        }],
        current_speaker="Teacher",
        round=0,
        max_rounds=3,
        terminated=False
    )
    print("\n=== 初始状态已设置 ===")
    print("系统: 让我们开始关于人工智能是否应该在教育领域全面应用的课堂讨论。请教师主持讨论。\n")

    # 运行图
    print("=== 开始运行辩论 ===")
    for output in graph.stream(initial_state):
        if isinstance(output, dict):
            if "state" in output:
                state = output["state"]
                if state["messages"]:
                    latest_message = state["messages"][-1]
                    print(f"轮次: {state['round']}, 发言者: {latest_message['speaker']}")
                    print(f"内容: {latest_message['content']}\n")
                    print("-" * 50)

if __name__ == "__main__":
    main()
