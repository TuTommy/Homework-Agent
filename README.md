# SmartAgents 多智能体课堂辩论系统

这是一个基于 LangGraph 和通义千问的多智能体课堂辩论项目，讨论主题为“人工智能是否应该在教育领域全面应用”。

## 功能

- 7 个角色明确的 Agent：教师主持人、正方 3 人和反方 3 人。
- 使用 LangGraph 固定编排辩论流程，教师总结后结束。
- Con1 采用 ReAct：调用工具、读取观察结果、再组织反方回应。
- 短期记忆读取最近 5 条辩论消息；长期记忆保存教师总结并在下一轮读取。
- 提供资料检索工具与辩论评分计算工具。
- 自动保存完整辩论记录和工具调用日志。

## 项目结构

```text
code/
  main.py                 # 项目入口
  requirements.txt        # Python 依赖
  README.md               # 详细运行说明
  .env.example            # API Key 配置模板
  long_term_memory.json   # 长期记忆示例
  outputs/                # 一份运行记录和工具日志
实验报告_多智能体课堂辩论系统.pdf
```

## 快速开始

进入 `code` 文件夹后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填入自己的 DashScope API Key，再运行：

```powershell
python main.py
```

`.env` 不会被 Git 跟踪，请不要提交真实 API Key。
