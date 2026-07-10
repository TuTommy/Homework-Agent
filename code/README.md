# 多智能体课堂辩论系统

## 项目主题

本项目使用 LangGraph 构建多智能体课堂辩论系统，讨论主题是：

> 人工智能是否应该在教育领域全面应用？

## Agent 分工

- Teacher：主持辩论、开场并在最后总结。
- Pro1：从教学效率角度支持 AI。
- Pro2：从个性化学习角度支持 AI。
- Pro3：从教育创新和资源公平角度支持 AI。
- Con1：使用 ReAct 和工具调用，从证据角度反驳。
- Con2：从师生关系和教育本质角度反驳。
- Con3：从学习主体性、隐私和公平风险角度反驳。

## 技术功能

- 使用 LangGraph 控制辩论流程。
- 使用短期记忆：读取最近 5 条辩论消息。
- 使用长期记忆：保存并读取教师总结。
- 使用两种工具：资料检索工具和辩论评分计算工具。
- Con1 使用 ReAct：工具调用、读取结果、生成回应。
- 自动保存完整辩论记录和工具调用记录。

## 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt