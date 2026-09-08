# Senza (森座)

> **生产级 Agent 运行时 — Rust 性能，Python 易用，崩溃可恢复，成本可控**

Senza 是 oh-my-harness Rust runtime 的 Python SDK，基于 PyO3 构建。面向需要长流程编排、崩溃恢复和成本管控的生产级 AI Agent 场景。

### 核心卖点

| 特性 | 说明 |
|------|------|
| ⚡ **Rust 内核** | PyO3 绑定，比纯 Python 框架更高的吞吐和更低的内存占用 |
| 🛡️ **原生崩溃恢复** | 工作流持久化 + 断点恢复，长流程不丢失进度 |
| 💰 **内置预算管控** | 定价感知 + 预算上限 + 超限回调，每一分钱都看得见 |
| 🔧 **两层 API** | Agent 层（单轮对话/工具调用/流式）+ Runtime 层（多步工作流/条件路由/暂停取消） |
| 🧠 **知识与记忆** | 本地知识源 RAG、记忆写入/删除接口、会话召回装配接口 |

### Showcase

两个完整应用 demo，不是 toy example：

| 项目 | 场景 | 展示能力 |
|------|------|---------|
| [**blender-scene-generator**](https://github.com/oh-my-harness/blender-scene-generator) | 自然语言 → Blender 3D 场景 | AgentHarness + WorkflowEngine + human-in-the-loop |
| [**eda-studio**](https://github.com/oh-my-harness/eda-studio) | LLM 驱动 RTL→GDS 芯片设计全流程 | 长流程编排 + 崩溃恢复 + 失败回环路由 + 多工具协调 |

![Blender demo](https://raw.githubusercontent.com/oh-my-harness/blender-scene-generator/main/docs/examples/rainy_neon_alley.png)

### 系统学习

如果希望先理解 Agent Core、14 个 Hook、Plugin 装配和能力边界，再开始拼 API，可以按顺序阅读
[《从 Agent 理论到 Senza 实践》](academy/textbook/README.md)，并配合
[Senza Academy 十个实验](academy/README.md)运行。教材使用《动手学 AI Agent》的理论问题作为
学习坐标，但工程组件、源码导读和演示均替换为当前 Runtime/Senza 实现。
Academy recorded Lab、live example 和严格 layer test 的统一方向见
[场景统一计划]()。

### 与其他框架对比

| 特性 | Senza | LangGraph | CrewAI | AutoGen |
|------|-------|-----------|--------|---------|
| 实现语言 | Rust 内核 + Python SDK | 纯 Python | 纯 Python | 纯 Python |
| 崩溃恢复 | ✅ 原生持久化 + 断点恢复 | ❌ 需自建 checkpoint | ❌ | ❌ |
| 预算管控 | ✅ 内置定价 + 预算上限 | ❌ | ❌ | ❌ |
| 工作流编排 | ✅ 条件路由/暂停/取消 | ✅ 图编排 | ✅ 顺序为主 | ✅ 对话编排 |
| 生产级 demo | ✅ 芯片设计 RTL→GDS | ❌ | ❌ | ❌ |
| 流式输出 | ✅ 原生 async | ✅ | ❌ | ✅ |

---

## 安装

```bash
pip install senza-sdk
```

```python
import senza

print(senza.version())  # e.g. "1.2.0"
```

---

## 快速上手

### 何时用 Agent，何时用 Workflow？

**简单判断**：一个 prompt + 几个工具能完成 → 用 Agent。多个 prompt 串联、条件分支或需要持久化 → 用 Workflow。

| 场景 | 用什么 |
|------|--------|
| 单轮问答 / 工具调用 | `AgentHarness` |
| 多步流程、条件分支 | `WorkflowEngine` |
| 人工介入 / 暂停恢复 | `WorkflowEngine` |
| 崩溃恢复 | `WorkflowEngine` + `with_task_store` |
| 预算管控 | 两者皆可（Agent `.budget()`，Workflow `.with_pricing()`）|

### Agent 示例

```python
import senza

provider = senza.providers.openai(api_key="sk-...")

harness = (
    senza.HarnessBuilder("gpt-4o")
    .provider("*", provider)
    .system_prompt("你是一个有用的助手。")
    .max_tokens(512)
    .build()
)

print(harness.chat("用一句话解释闭包。"))
```

`harness.chat(text)` 是 1.2.0 新增的便捷方法，内部等价于
`senza.extract_text(harness.prompt_and_collect(text))` —— 一步取回纯文本回复。
如需逐 token 流式或原始事件，用 `stream_prompt()` 或 `prompt_and_collect()`。

### 多模态附件（图片 / 文档）

`prompt` / `chat` / `prompt_and_collect` / `steer` / `follow_up` / `next_turn`
均支持 `attachments=` 参数（1.2.4+）：

```python
harness.chat(
    "描述这张图",
    attachments=[senza.image_url("https://example.com/i.png")],
)
harness.chat("总结这份文档", attachments=[senza.document_file("report.pdf")])
```

构造函数：`image_url(url)`、`image_base64(data, mime_type="image/png")`、
`document_url(url, name=None)`、`document_file(path, name=None)`。
端点需支持对应模态（图片 / 文档），否则 provider 返回 400。

> **⚠️ `senza.Agent`（test-utils mock）的 `prompt(attachments=...)` 语义注意**：
> 带附件时底层走 `prompt_with_messages`——**整个 transcript 被替换**；
> 不带附件时是追加。生产路径 `AgentHarness.prompt()` 始终追加，不受影响。
> 带附件的多轮对话请优先使用 `AgentHarness`。

### Workflow 示例

```python
import senza

provider = senza.providers.openai(api_key="sk-...")

workflow = {
    "entry_step": "writer",
    "steps": [
        {"id": "writer", "name": "写作", "prompt": "写一句关于猫的故事。", "allowed_tools": []},
        {"id": "reviewer", "name": "审阅", "prompt": "给这个故事打分 1-5。", "allowed_tools": []},
    ],
    "edges": [{"from": "writer", "to": "reviewer"}],
}


def judge(ctx):
    if ctx["step_id"] == "writer":
        return "to:reviewer"
    return "done"


engine = senza.WorkflowEngine(
    workflow, provider, "gpt-4o", senza.create_judge(judge)
).with_max_tokens(256)

engine.run()

for record in engine.step_history():
    r = record.get("result")
    print(f"{record['step_id']}: {r['output'][:80] if r else '(无结果)'}")
```

> **Judge 返回值**：`"to:<step_id>"` 跳转 / `"retry"` 重跑 / `"fail:<reason>"` 失败 / `"done"` 结束。详见 [API 参考](docs/api-reference.md#judge)。

---

## 指南

### Provider 配置

`senza.providers.openai` 支持 `base_url` 参数，任何兼容 OpenAI Chat Completions API 的服务都能直接接入（通义千问、DeepSeek、Ollama 等）。见 [Provider 配置指南](docs/providers.md)。

### 崩溃恢复

```python
import tempfile

with tempfile.TemporaryDirectory() as store_dir:
    engine = senza.WorkflowEngine(
        workflow, provider, "gpt-4o", senza.create_judge(judge)
    ).with_task_store(store_dir)
    task_id = engine.task_id()
    engine.run()

    # 崩溃后恢复
    restored = senza.WorkflowEngine.restore(
        store_dir, task_id, provider, "gpt-4o", senza.create_judge(judge)
    )
    print(restored.state(), restored.current_step())
```

### 流式输出

```python
import asyncio
import senza


async def main():
    provider = senza.providers.openai(api_key="sk-...")
    harness = senza.HarnessBuilder("gpt-4o").provider("*", provider).max_tokens(256).build()
    async for event in senza.stream_prompt(harness, "用一句话解释闭包。", timeout_ms=30000):
        if event["type"] == "text_delta":
            print(event.get("text", ""), end="", flush=True)


asyncio.run(main())
```

> `stream_prompt` / `stream_events` / `stream_run` 是模块级 async generator，不是 `AgentHarness` 的方法。

### 内置文件工具

```python
harness = (
    senza.HarnessBuilder("gpt-4o")
    .provider("*", provider)
    .plugin(senza.create_fs_tools_plugin())  # bash/read/write/edit/grep/glob
    .env(senza.create_os_env("."))  # 真实文件系统 + shell
    .build()
)
```

---

### 策略插件

`senza.strategy` 提供 10 个策略 Plugin 工厂和 2 个辅助函数，覆盖安全防护、循环断路、审计日志、注入检测等场景：

```python
harness = (
    senza.HarnessBuilder("gpt-4o")
    .provider("*", provider)
    .plugin(senza.strategy.safety_defaults())  # bash 黑名单 + 路径穿越防护
    .plugin(senza.strategy.loop_safety())  # 死循环/重复/连续失败断路器
    .build()
)
```

### 知识与记忆

给 Agent 挂载本地知识源（RAG）：

```python
# 本地知识源 RAG
docs = senza.knowledge.local_source(
    path="/data/wiki",
    source_id="wiki",
)
knowledge = senza.knowledge.plugin(sources=[docs])

harness = (
    senza.HarnessBuilder("gpt-4o")
    .provider("*", provider)
    .plugin(knowledge)  # LLM 可调用 knowledge_search / knowledge_read
    .build()
)
```

> Memory API 还提供 `memory_write` / `memory_forget`。当前内置 `memory_store()` 是进程内演示实现，不持久化；Session Recall 已暴露 repo/index/source/plugin 装配接口，召回前需确保索引已有数据。

### 子 Agent 派发

通过 `enable_spawn()` 在 Agent 上启用子 Agent 派发能力。启用后，LLM 可调用 `spawn_agent`、`await_subagent_reply`、`query_subagent` 等工具进行多 Agent 协作：

```python
harness = (
    senza.HarnessBuilder("gpt-4o")
    .provider("*", provider)
    .enable_spawn(
        model="gpt-4o",
        provider=provider,
        session_dir="/tmp/sessions",
    )
    .system_prompt("你是任务协调者，派发子 Agent 并行处理子任务。")
    .build()
)
```

> `enable_spawn` 为主 Agent 注册 MessageBus 和 5 个管理 tool：`spawn_agent`、`message_subagent`、`await_subagent_reply`、`query_subagent`、`abort_subagent`。Runtime 还定义 2 个可由 child plugin 贡献的子 Agent 侧 tool（`message_main`、`await_main_message`），但当前 Senza child factory 使用 `NoopPlugin`，不会自动挂载它们，也不会递归 spawn。spawn 是异步的——`spawn_agent` 立即返回 `agent_id`，子 Agent 完成后结果自动注入主对话。

## 示例（Live Tests）

当前 40 个 live/API 示例脚本位于
[`live-tests/examples/`](live-tests/examples/)：23 个运行时同名镜像
（`01_prompt_streaming` … `23_infra_integration`）+ 17 个原仓库根示例
（`30` … `46`）。

```bash
# P1 统一入口：Catalog 检索、依赖诊断与 legacy implementation 运行
python -m academy.scenarios list
python -m academy.scenarios describe agent.tool_calling
python -m academy.scenarios doctor agent.tool_calling
python -m academy.scenarios run agent.tool_calling
python -m academy.scenarios course 01 --mode recorded
python -m academy.scenarios course 01 --mode live

# 当前路径继续兼容
source ~/.omp_llm_env && python live-tests/examples/01_prompt_streaming.py
python live-tests/examples/30_basic_prompt.py             # 无 key → 打印 SKIP 并 exit 0
```

`live-tests/` 另含按架构层组织的**真实 LLM 集成测试**（agent / loop / tools / runtime /
strategy），镜像 runtime 仓库的 `llm-harness-live-tests` 惯例；每层含一个不依赖 key 的
离线构造冒烟。它将继续承担严格行为验证，不会被可运行文档的弱断言替代。详见
[`live-tests/README.md`](live-tests/README.md)。

```bash
python -m pytest live-tests/ -v                           # 跑 5 层测试（真实 DeepSeek）
```

> P1 已实现 Single Scenario Catalog、Runner 和 Academy manifest bridge；当前 `run`
> 仍执行 catalog 指向的 legacy script。native scenario adapters、统一 result envelope 和
> strict verifier 尚未实现。`live-tests/examples/` 将作为 legacy adapter 与 source pool，
> 旧脚本路径在兼容窗口内继续工作；完整迁移步骤见
> [场景统一计划]()。

---

## API 结构

Senza 的公开 API 分两层：

- **顶层高频 API**：`HarnessBuilder`、`create_tool`、`create_judge`、
  `create_plugin`、`create_fs_tools_plugin`、`create_os_env` 等 —— 每个Agent都会用到的函数。
- **子模块分组**：较低频 API 按领域组织：
  - `senza.providers` — LLM 提供商工厂（`openai`、`anthropic`）
  - `senza.hooks` — 14 个生命周期 hook 工厂
  - `senza.strategy` — 10 个策略 Plugin 工厂 + 2 个辅助函数
  - `senza.knowledge` — 知识源、记忆和会话召回装配工厂
  - `senza.rules` — 规则链和谓词工厂
  - `senza.infra` — 审计 sink、trace exporter、sandbox 工厂

完整 API 速查（含所有方法签名、事件类型、judge ctx 字段、hooks、rules 等）见 [docs/api-reference.md](docs/api-reference.md)。

## 工具创建

### 用 `@senza.tool` 装饰器创建工具

创建工具的推荐方式是使用 `@senza.tool` 装饰器，它从类型提示自动推导 JSON Schema：

```python
import senza


@senza.tool
def search(query: str) -> str:
    """搜索网络信息。"""
    # 实现...
    return results
```

函数名成为工具名，docstring 成为描述，类型注解定义参数 schema。同步和异步函数均支持。

### 用 `create_tool` 手动创建工具

```python
tool = senza.create_tool(
    name="search",
    description="搜索网络信息",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    callback=lambda args, ctx: {"content": [{"type": "text", "text": "结果"}], "terminate": False},
)
```

`parameters` 接受 dict 或 JSON 字符串。回调签名可以是 `(args, ctx)` 或仅 `(args)`。

## 设计文档

见 [`SENZA_DESIGN.md`](SENZA_DESIGN.md) — 完整架构、缺口分析、路线图。

## 开发

开发 Senza 本身见 [DEVELOPMENT.md](DEVELOPMENT.md)——涵盖本地搭建、测试（`./scripts/cargo_checks.sh` 一键跑 fmt+clippy+cargo test+pytest）、发布流程、CI 行为。

## 贡献

欢迎参与！见 [CONTRIBUTING.md](CONTRIBUTING.md) — 涵盖开发环境搭建、测试方法、PR 规范和 good first issue 指引。
