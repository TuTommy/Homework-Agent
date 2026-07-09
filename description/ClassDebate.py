from typing import List, Dict, Any, Annotated, Sequence, Literal, TypedDict, Union
from datetime import datetime
import operator
import json

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph
from langchain_core.tools import tool

# 配置模型
model = ChatOpenAI(
    model="qwen-turbo",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-422a30c66061427cbc47d89a5712540d",
    temperature=0.7
)

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
        memory_str = "\n".join([f"{m['speaker']}: {m['content']}" for m in self.memory[-5:]])
        return f"""
{self.system_prompt}

历史对话：
{memory_str}

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
    DebateAgent(f"Pro{i}", "supporter", 
        "你是支持在教育领域全面应用人工智能的一方。请基于效率提升、个性化学习和教育创新的角度进行论述。")
    for i in range(1, 4)
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
for i in range(2, 4):
    con_team.append(
        DebateAgent(f"Con{i}", "opponent",
            "你是反对在教育领域全面应用人工智能的一方。请基于教育本质、师生关系和学习体验的角度进行论述。")
    )

# 定义节点函数
def teacher_node(state: AgentState) -> Dict:
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
    
    if "TERMINATE" in response.content or state["round"] >= state["max_rounds"]:
        return {"state": state, "next": None}
    
    # 教师只在开始时发言，然后转给第一位正方
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

def con_node(state: AgentState, agent: DebateAgent) -> Dict:
    """反方辩手节点处理函数"""
    context = agent.get_context(state["messages"])
    
    # 为Con1添加ReAct流程
    if agent.name == "Con1":      
        # 拿到搜索的结果，这里只做一个示例，实际应用中需要llm自己调用工具拿到工具信息。
        search_result = search_tool.invoke("ai concerns teacher impact")

        # 回应
        react_prompt = f"""{context}
搜索结果：{search_result}
请基于以上搜索结果，从教育本质和师生关系角度论证为什么不应该在教育领域全面应用人工智能。
"""
        response = model.invoke([HumanMessage(content=react_prompt)])
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
def build_debate_graph() -> Graph:
    """构建辩论流程图"""
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("teacher", teacher_node)
    for i, agent in enumerate(pro_team):
        workflow.add_node(f"pro{i+1}", lambda state, a=agent: pro_node(state, a))
    for i, agent in enumerate(con_team):
        workflow.add_node(f"con{i+1}", lambda state, a=agent: con_node(state, a))
    
    # 设置顺序边
    # Teacher -> Pro1 -> Con1 -> Pro2 -> Con2 -> Pro3 -> Con3 -> Teacher
    workflow.add_edge("teacher", "pro1")
    workflow.add_edge("pro1", "con1")
    workflow.add_edge("con1", "pro2")
    workflow.add_edge("pro2", "con2")
    workflow.add_edge("con2", "pro3")
    workflow.add_edge("pro3", "con3")
    workflow.add_edge("con3", "teacher")
    
    # 设置入口点
    workflow.set_entry_point("teacher")
    
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
