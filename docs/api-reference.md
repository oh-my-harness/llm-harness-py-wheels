# Senza API 参考

## Provider

```python
senza.providers.openai(
    api_key,
    base_url=None,
    chat_path=None,
    thinking_scheme=None,
    parse_reasoning_content=True,
    tolerant_keepalive=True,
)
senza.providers.anthropic(api_key, base_url=None, messages_path=None)
```

> 接入通义千问 / DeepSeek / Ollama 等 OpenAI 兼容模型？见 [Provider 配置指南](providers.md)。

## Agent 层

### HarnessBuilder

| 方法 | 说明 |
|------|------|
| `HarnessBuilder(model)` | 创建 builder |
| `.provider(pattern, provider)` | 注册 LLM provider（glob 匹配模型名，`"*"` 匹配所有） |
| `.system_prompt(text)` | 设置系统提示 |
| `.max_tokens(n)` / `.temperature(t)` | LLM 参数 |
| `.thinking_level(level)` | 设置 thinking level |
| `.auto_compact(b)` / `.compaction_reserve_tokens(n)` / `.compaction_keep_recent_tokens(n)` | Compaction 配置 |
| `.compaction_model(model, context_window, max_tokens)` | 独立 compaction 模型 |
| `.compaction_prompt(system_prompt=None, user_template=None)` | 自定义 compaction 提示词 |
| `.compaction_query(query=None)` | 查询聚焦 compaction |
| `.should_stop_hook(hook)` / `.hooks([hook, ...])` | 注册 ShouldStopHook / 批量 hooks |
| `.after_turn_hook(hook)` | 注册 AfterTurn hook（便捷方法） |
| `.final_answer_validator(hook)` | 注册最终回答校验 hook（提交前接受或拒绝） |
| `.retry(max_retries, base_delay_ms)` | 瞬时错误重试配置 |
| `.model_info(context_window, max_tokens)` | 模型元数据 |
| `.final_answer_mode("heuristic"\|"tool")` | 最终回答判定模式 |
| `.stream_options(timeout_ms, max_retries)` | 流式请求选项 |
| `.queue_capacity(n)` | steer/follow-up 队列容量 |
| `.budget(limit, exceeded_hook=None)` | 预算上限 + 超限回调 |
| `.pricing(provider)` | 定价 provider（成本计算） |
| `.usage_ledger(ledger)` | 共享 UsageLedger（多 Agent 成本汇总） |
| `.skill(skill)` / `.skills([skill, ...])` | 注册 skill(s) |
| `.disable_skill_read_tool()` | 关闭 SkillReadTool 自动注册 |
| `.response_format(fmt)` | 设置 JSON response format（`create_json_object_format()` / `create_json_schema_format()`） |
| `.knowledge_access(scope, principal, kind)` | 知识访问控制（scope/principal/kind） |
| `.mcp_server(name, config)` | 注册 MCP server（stdio/HTTP/SSE） |
| `.mcp_config_file(path)` | 从配置文件加载 MCP servers |
| `.with_mcp_manager(manager)` | 注入预配置的 McpManager |
| `.session_repo(repo, session_id=None)` | 设置会话持久化仓库 |
| `.tool(tool)` / `.plugin(plugin)` | 注册工具/插件 |
| `.tools([tool, ...])` | 批量注册工具（等价多次 `.tool()`） |
| `.env(env)` | 设置执行环境（`create_os_env(...)`），启用 bash/read/write/edit/grep/glob 工具 |
| `.enable_spawn(model, provider, session_dir, max_concurrent=None)` | 启用子 Agent 派发（主 Agent 注册 MessageBus + 5 个管理 tool；`max_concurrent` 限制并发子 Agent 数，`None` 不限） |
| `.build()` | 返回 `AgentHarness` |

### AgentHarness

| 方法 | 说明 |
|------|------|
| `.prompt_and_collect(text, timeout_ms=30000, attachments=None)` | 发送提示并收集事件（推荐）；`attachments` 为多模态附件列表（1.2.4+） |
| `.chat(text, timeout_ms=30000, attachments=None)` | 发送提示并返回拼接后的纯文本回复（`str`） |
| `.chat_async(text, timeout_ms=30000, attachments=None)` | `chat()` 的非阻塞 async 版本（线程池执行） |
| `.prompt(text, attachments=None)` | 发送提示（阻塞，需配合线程收集事件） |
| `.prompt_async(text, timeout_ms=30000, attachments=None)` | `prompt_and_collect()` 的非阻塞 async 版本 |
| `.collect_until_settled(timeout_ms=30000)` | 收集事件直到完成 |
| `.events(timeout_ms=5000)` | 流式事件迭代器 |
| `.inspect()` | 返回 harness 内部状态快照（dict） |
| `.set_model(model, context_window=None, max_tokens=None)` | 运行时切换模型 |
| `.set_system_prompt(text)` | 修改系统提示 |
| `.set_temperature(t)` | 修改温度 |
| `.set_thinking_level("high")` | "off"/"minimal"/"low"/"medium"/"high"/"xhigh"/"budget:N" |
| `.set_max_tokens(n)` | 修改最大输出 token 数 |
| `.set_tools(tools)` | 替换工具集 |
| `.set_active_tools(tools)` | 限定下一轮工具子集（传 `None` 恢复全部） |
| `.steer(text, attachments=None)` / `.follow_up(text, attachments=None)` | 运行中注入消息（可带多模态附件；Idle 阶段静默丢失） |
| `.next_turn(text, attachments=None)` | 开启下一轮对话 |
| `.continue_run()` | 继续运行（配合 steer/follow_up） |
| `.compact()` | 手工触发 compaction，返回 `tokens_before` / `tokens_after` / `compressed_entries` |
| `.session_metadata()` | 会话元数据 dict（`id` / `name` / `created_at` / `updated_at` / `model` / ...） |
| `.usage()` | 查询成本统计 |
| `.usage_ledger()` | 返回 UsageLedger 快照（dict） |
| `.reset_usage()` | 重置成本统计 |
| `.message_count()` | 当前消息数 |
| `.phase()` | 当前阶段：`"idle"` / `"turning"` / `"compacting"` / `"branching"` |
| `.get_messages()` | 获取完整对话历史 |
| `.last_response()` | 获取最近一条 assistant 回复文本 |
| `.abort()` | 取消当前提示 |
| `.wait_for_idle()` / `.wait_for_settled()` | 阻塞等待 idle/settled |
| `.clear_steering_queue()` / `.clear_follow_up_queue()` | 清空特定队列 |
| `.clear_all_queues()` / `.has_queued_messages()` | 队列管理 |
| `.fork_branch()` / `.list_branches()` / `.navigate_tree()` | 会话分支管理 |
| `.read_active_path()` / `.read_all_entries()` | 读取会话历史 |
| `.delete_branch()` / `.generate_branch_summary()` | 分支删除与摘要 |
| `.shutdown()` | 释放底层资源 |
| `__enter__` / `__exit__` | Context manager 支持 |

### 事件类型

Terminal: `settled`, `aborted`, `error`.

Streaming: `text_delta` (has `.text`), `message_end`, `tool_call_start`, `tool_call_end`, `tool_execution_start`, `tool_execution_end`, `thinking_delta`.

Harness: `phase_change`, `compaction_start`, `compaction_end`, `tools_update`.

事件类型字符串也可通过 `senza.EventType` 常量引用（避免拼写错误）：
`TEXT_DELTA`, `THINKING_DELTA`, `MESSAGE_END`, `TOOL_CALL_START`, `TOOL_CALL_END`,
`TOOL_RESULT`, `SETTLED`, `ABORTED`, `AGENT_END`, `ERROR`, `WORKFLOW_DONE`, `WORKFLOW_FAILED`。

### 三种 prompt 方式

| 方式 | 适用场景 | 说明 |
|------|---------|------|
| `harness.prompt_and_collect(text)` | **推荐**，同步场景 | 一步发送 + 收集所有事件，返回 `list[dict]` |
| `senza.stream_prompt(harness, text)` | 需要流式输出 | 模块级 async generator，逐 token yield 事件 |
| `harness.prompt(text)` + `harness.events()` | 需要线程级控制 | prompt 阻塞，需另起线程收集事件 |

### 工具创建

```python
tool = senza.create_tool(
    name="search",
    description="Search the web",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    callback=lambda args, ctx: {
        "content": [{"type": "text", "text": f"Results for {args['query']}"}],
        "terminate": False,
    },
)
```

- `parameters` / `parameters_schema`: JSON Schema，接受 dict 或 JSON 字符串。`parameters` 是推荐名称。
- `callback`: `(args: dict, ctx: ToolContext) -> dict`。`args` 是**完整的参数字典**（如 `{"query": "cats"}`），回调内自行用 `args["query"]` 取值。`ctx` 可选——函数接受 2 参时传 `ctx`，1 参时只传 `args`。
  - ⚠️ **回调签名不是独立参数**：`def search(query: str)` 是**错误**的——`query` 会收到整个 dict 而非字符串。正确写法是 `def search(args: dict, ctx=None)` 然后内部 `args["query"]`，或使用下面的 `@senza.tool` 装饰器。
- 返回 dict: `{"content": [ContentBlock...], "terminate": bool}`。`terminate=True` 停止 agent 循环。也接受纯字符串（自动包装为 text content）或不含 `content` 键的 dict（整体 JSON 序列化为 text）。
- **多模态返回（1.2.4+）**：回调可返回 `Attachment`（裸值）、含 `Attachment` 的 list/tuple（元素为 `Attachment` 或 str）、或在 `content` 列表中混入 `Attachment`——自动转为 image/document 内容块，多模态模型可直接消费。
- `create_tool(..., report_duration=True)`（1.2.4+）：在回传 LLM 的结果末尾附加执行耗时标注（如 `[duration: 812ms]`），让模型感知慢操作。默认关闭；仅当工具经 agent loop 的 HookedTool 包装时生效。

#### `@senza.tool` 装饰器（便捷方式）

用类型提示自动生成 JSON Schema，回调按独立参数接收值：

```python
@senza.tool
def search(query: str, limit: int = 10) -> str:
    """Search the web."""
    return f"Results for {query} (top {limit})"

# 等价于 create_tool + 自动 schema + 自动 kwargs 解包
```

- 类型提示 → JSON Schema（`str→string`, `int→integer`, `float→number`, `bool→boolean`）
- 无默认值的参数自动标记为 `required`
- 支持 `async def`
- docstring 自动作为工具描述

### 多模态附件（1.2.4+）

```python
a1 = senza.image_url("https://example.com/i.png")
a2 = senza.image_base64(raw_bytes, mime_type="image/jpeg")  # bytes 自动 base64
a3 = senza.document_url("https://example.com/d.pdf")        # 按扩展名推断 media_type
a4 = senza.document_file("./report.pdf", name=None)         # 读本地文件

harness.chat("描述这张图", attachments=[a1])
```

| 构造函数 | 说明 |
|---------|------|
| `senza.image_url(url)` | 公网图片 URL |
| `senza.image_base64(data, mime_type="image/png")` | 内联图片，`bytes` 自动编码 |
| `senza.document_url(url, name=None)` | 文档 URL（media_type 按扩展名推断，未知扩展名报错） |
| `senza.document_file(path, name=None)` | 本地文档（`.pdf`/`.txt`，读入内存） |

端点需支持对应模态，否则 provider 返回 400。`Attachment` 对用户不透明；
可作为工具回调返回值（见上）。`AgentHarness.get_messages()` 返回的 content
含完整的 `{"type": "image"/"document", ...}` 块。

> ⚠️ `senza.Agent`（test-utils mock）的 `prompt(attachments=...)` 带附件时
> **替换整个 transcript**（底层 `prompt_with_messages`）；不带附件时追加。
> 生产路径 `AgentHarness.prompt()` 始终追加。带附件多轮对话用 `AgentHarness`。

### 内置 fs 工具

```python
harness = (
    senza.HarnessBuilder("gpt-4o")
    .provider("*", provider)
    .plugin(senza.create_fs_tools_plugin())  # bash/read/write/edit/grep/glob
    .env(senza.create_os_env("."))  # 真实文件系统 + shell
    .build()
)
```

## Runtime 层

### WorkflowEngine

| 方法 | 说明 |
|------|------|
| `WorkflowEngine(workflow_dict, provider, model, judge, session_base_dir="sessions", env=None)` | 构造引擎；`env` 传 `create_os_env(...)` 以启用 shell 执行 |
| `.with_tool(tool)` | 注册工具 |
| `.with_external_tool(tool)` | 注册 WaitForExternalEventTool（人工介入） |
| `.with_executor(name, exec)` | 注册命名执行器 |
| `.with_hooks([hooks])` | 注册 hooks |
| `.with_step_plugin(step_id, plugin)` | per-step 注入 plugin |
| `.with_step_builder(step_id, customize)` | per-step builder 定制闭包（覆盖共享设置，如 system_prompt） |
| `.with_task_store(dir)` | 启用持久化 |
| `WorkflowEngine.list_tasks(task_store_dir)` | 类方法 — 列出已持久化的任务 |
| `.with_max_tokens(n)` / `.with_thinking_level(level)` | per-step LLM 参数（共享，所有 step） |
| `.with_max_steps(n)` / `.with_max_retries(n)` | 总步数上限 / per-step 连续 Retry 上限（超限 → Failed） |
| `.with_pricing(provider)` | 定价 provider |
| `.set_context_variable(key, value)` | 设置共享上下文变量 |
| `.get_context_variable(key)` | 读取共享上下文变量 |
| `.run()` | 执行（阻塞） |
| `.run_async(timeout_ms=300000)` | 非阻塞 async 版本（线程池执行） |
| `.state()` | "idle"/"running"/"paused"/"succeeded"/"failed"/"cancelled" |
| `.current_step()` / `.step_history()` | 进度查询 |
| `.task_id()` | 任务 ID（`"task-<uuid>"`） |
| `.pause(reason)` / `.resume()` / `.cancel(reason)` | 流程控制 |
| `WorkflowEngine.restore(store_dir, task_id, provider, model, judge)` | 类方法 — 崩溃恢复 |
| `WorkflowEngine.restore_from_step(store_dir, task_id, step, provider, model, judge)` | 类方法 — 从指定 step 恢复 |
| `.checkpoint(desc, payload)` / `.total_cost()` | 检查点 & 成本 |
| `.inspect()` | 返回引擎内部状态快照（dict） |
| `.subscribe(timeout_ms=5000)` | 事件流迭代器 |
| `__enter__` / `__exit__` | Context manager 支持 |

### Workflow Dict Schema

```python
{
    "entry_step": "step1",  # must be in steps
    "steps": [...],  # list of step dicts
    "edges": [...],  # list of edge dicts
}
```

Step 类型由引擎自动检测：**有 `"executor"` key → Executor step；否则 → LLM step。**

| 字段 | LLM | Executor | 类型 |
|-------|:---:|:--------:|------|
| `id` | ✅ | ✅ | str (unique) |
| `name` | ✅ | ✅ | str |
| `prompt` | ✅ | — | str |
| `allowed_tools` | ✅ | — | str[] (empty = no tools) |
| `structured` | ✅ | — | bool (设 `true` 启用 JSON 提取) |
| `executor` | — | ✅ | str (registry key) |
| `executor_config` | — | ✅ | dict (optional) |

### Edges

```python
{"from": "step1", "to": "step2"}  # unconditional
{"from": "step1", "to": "step2", "condition": "pass"}  # label (judge interprets)
{
    "from": "step1",
    "to": "step2",
    "condition": {"op": "eq", "pointer": "/status", "value": "ok"},
}  # declarative
```

### 声明式 ConditionExpr

| op | params | semantics |
|----|--------|-----------|
| `exists` | `pointer` | path exists in structured |
| `missing` | `pointer` | path does not exist |
| `eq` | `pointer`, `value` | equals |
| `ne` | `pointer`, `value` | not equals |
| `gt` / `gte` / `lt` / `lte` | `pointer`, `value`(float) | numeric comparison |

`pointer` uses RFC 6901 JSON Pointer (e.g. `/status`, `/data/0/score`).

**Auto-enable**: if any edge has an Expr condition and judge is NoopJudge, engine auto-switches to built-in `EdgeConditionJudge`.

### Judge

```python
def my_judge(ctx: dict) -> str:
    structured = ctx.get("structured") or {}
    if structured.get("status") == "ok":
        return "to:next_step"
    elif structured.get("retry_needed"):
        return "retry"
    else:
        return "fail:quality gate failed"


judge = senza.create_judge(my_judge)
```

返回值编码：

| 返回值 | 含义 |
|--------|------|
| `"to:<step_id>"` | 跳转到指定 step |
| `"retry"` | 重跑当前 step（计入 retry_count） |
| `"fail:<reason>"` | 标记工作流失败 |
| `"abort:<reason>"` | 结束工作流（视为成功完成） |
| `"done"` | 同 `abort:done`，结束工作流 |

### Judge ctx 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `step_id` | str | 当前步 ID |
| `output` | str | 当前步执行输出 |
| `structured` | dict \| None | 结构化结果（step 声明 `structured: true` 时，引擎从 final answer 提取的 JSON） |
| `structured_status` | str | `"not_required"` / `"ok"` / `"failed"`（结构化提取状态） |
| `step_count` | int | `step_history` 长度（含当前步） |
| `retry_count` | int | 当前 step 的连续 Retry 次数（0 = 首次执行后；与 `with_max_retries` 同口径） |
| `tool_calls_count` | int | 本步工具调用次数（不含 `submit_step_result`，该工具已移除） |

### Retry 语义

- `with_max_retries(n)`：**per-step** 连续 Retry 上限。`n=3` 允许 3 次 Retry，第 4 次触发 Failed（不含原始执行）。
- `with_max_steps(n)`：**workflow 级** 总步数护栏，含所有 Retry 重跑。超限 → Failed。
- judge 每次 Retry 后仍会被调用，engine 不自动吞重试——judge 自行决定是否继续 Retry。
- 与 `StepExecutionPolicy.max_attempts` 独立：最坏情况单步执行次数 = `max_retries × max_attempts`。
- 如需 per-回环 独立限制，在 judge 中读 `ctx["retry_count"]` 并自行决策。

### Executor

```python
senza.create_composite_judge()  # CompositeJudge（按节点注册独立路由）
senza.create_executor(callback)  # Python 回调执行器
senza.create_shell_executor(commands)  # Shell 命令执行器（命令白名单，需配合 create_os_env）
senza.create_http_executor(allowed_hosts)  # HTTP 调用执行器（host 白名单）
senza.create_fs_tools_plugin()  # bash/read/write/edit/grep/glob 六件套 Plugin（需配合 create_os_env）
senza.create_os_env(working_dir=".")  # OS 文件系统 + shell 执行环境（传给 WorkflowEngine(env=...)）
```

### Shared Context

```python
# Set before run
engine.set_context_variable("user_input", "hello")


# Executor reads context
def my_executor(ctx):
    user_input = ctx["context"]["user_input"]
    return {"output": f"Processed: {user_input}"}
```

### WorkflowEvent 类型 (subscribe)

| type | fields |
|------|--------|
| `step_started` | `step_id`, `step_name` |
| `step_finished` | `step_id`, `output`, `structured`, `tool_calls_count` |
| `paused` | `reason` |
| `resumed` | — |
| `cancelled` | `reason` |
| `failed` | `error` |

## Hooks（15 种）

```python
senza.hooks.before_turn(cb)  # cb(ctx: dict) -> None
senza.hooks.after_turn(cb)  # cb(ctx: dict) -> None
senza.hooks.before_run(cb)  # cb(ctx: dict) -> None
senza.hooks.after_provider_response(cb)  # cb(ctx: dict) -> None
senza.hooks.before_provider_request(cb)  # cb(ctx: dict) -> None
senza.hooks.before_tool_call(cb)  # cb(ctx: dict) -> str | dict  # allow/modify/deny
senza.hooks.after_tool_call(cb)  # cb(ctx: dict) -> str | dict
senza.hooks.should_stop(cb)  # cb(ctx: dict) -> bool
senza.hooks.before_compact(cb)  # cb(ctx: dict) -> Any
senza.hooks.transform_context(cb)  # cb(ctx: dict) -> dict
senza.hooks.prepare_next_turn(cb)  # cb(ctx: dict) -> Optional[dict]
senza.hooks.final_answer_validator(cb)  # cb(ctx: dict) -> None | str | dict  # 提交前接受/拒绝
senza.hooks.after_run(cb)  # cb() -> None  # run 结束后清理
senza.hooks.on_abort(cb)  # cb() -> None  # abort 时同步执行
senza.hooks.provider_error(cb)  # cb(ctx: dict) -> "retry" | "surface" | None
```

### provider_error hook

provider 非瞬态错误（重试耗尽后仍失败，如 text-only provider 拒绝含图片的
请求）上抛前调用。返回 `"retry"` 则同轮重试，返回 `"surface"` / `None`
则原样上抛。ctx 含 `run_id` / `started_at` / `turn_index` / `error` /
`context`（只读快照）/ `new_messages`（只读快照）。注册方式：
`builder.provider_error_hook(hook)`；多个 hook 按注册顺序执行，首个
retry 生效。Python 回调不回写历史——需要修复历史时用 preset hook
（`senza.strategy.vision_degrade()`）。

## Pricing

```python
senza.create_pricing_provider(table)  # 静态定价表 dict
senza.create_pricing_provider_callback(cb)  # cb(model, provider) -> dict | None
```

## Budget

```python
senza.create_budget_exceeded_hook(cb)  # cb(cost: dict, limit: float) -> bool
```

## Rules 审批

```python
senza.rules.contains(allowed)  # tool_name ∈ allowed
senza.rules.regex_field(arg_path, pattern)  # args[arg_path] 匹配正则
senza.rules.number_range(arg_path, min, max)  # 数值区间
senza.rules.rate_limit(max, window_seconds)  # 限流

chain = senza.rules.chain().rule("search", pred, "allow").fallback("deny").build()
hook = senza.rules.approval_hook(chain)  # → BeforeToolCallHook
```

## Skills

```python
senza.load_skills(path)  # 扫描目录下的 SKILL.md，返回 list[Skill]
```

## Event Channel（人工介入）

```python
handle, wait_tool = senza.create_event_channel("review-task")
# wait_tool 注册到 WorkflowEngine.with_external_tool(wait_tool)
# LLM 调用 wait_for_external_event 时暂停，直到 handle.submit() 被调用
handle.submit("approved", {"feedback": "Looks good!"})

# 审批门：approve/deny 语义 + 超时回落默认值（fail-safe）
handle, approval_tool = senza.create_human_approval_channel(
    "deploy-gate", timeout_seconds=300.0, default="deny",
)
# LLM 调用 request_human_approval 时暂停；应答只需提供 decision：
handle.submit("approve", {"decision": "approve"})

# 自由输入：LLM 提问，人给任意 JSON 值，超时返回默认值
handle, input_tool = senza.create_human_input_channel(
    "clarify-1", timeout_seconds=300.0, default=None,
)
handle.submit("42", {"value": 42})
```

说明：human channel 的 `handle.submit` 自动注入当前挂起请求的
`request_id`（调用方无需关心 `tool_use_id`）；tool 尚未发起请求时
submit 抛 `RuntimeError`。每个 channel 同一时刻支持一个挂起请求
（human-in-the-loop 的典型形态），不支持多并发挂起。

## Strategy（10 个 Plugin 工厂 + 2 个 preset hook + 2 个 helper）

```python
senza.strategy.safety_defaults() -> Plugin
senza.strategy.loop_safety(config: Optional[dict] = None) -> Plugin
senza.strategy.status_panel() -> Plugin
senza.strategy.memory_defense() -> Plugin

# MemoryDefense builder（自定义保护文件）:
senza.MemoryDefensePluginBuilder()
    .extra_file(name: str) -> MemoryDefensePluginBuilder
    .extra_files(names: list[str]) -> MemoryDefensePluginBuilder
    .build() -> Plugin

senza.strategy.injection_filter(patterns: Optional[list[str]] = None) -> Plugin
senza.strategy.source_tag(entries: list[dict]) -> Plugin
senza.strategy.project_instruction(
    env: ExecutionEnv, config: Optional[dict] = None,
) -> Plugin
senza.strategy.audit(
    sink_path: str, trace_id: Optional[str] = None, task_id: Optional[str] = None,
) -> Plugin
senza.strategy.notify() -> Plugin
senza.strategy.tool_output_guard(
    env: ExecutionEnv, config: Optional[dict] = None,
) -> Plugin

# preset hooks（返回 Hook，不是 Plugin）
senza.strategy.vision_degrade() -> Hook  # 注册到 builder.provider_error_hook()
senza.strategy.observation_shielding(config: Optional[dict] = None) -> Hook
# observation_shielding 注册到 builder.hooks([...])；config 键：
#   retained_turns: int = 5（保留最近 N 个 assistant turn 的观测）
#   placeholder: str（旧观测的替换文本）

# 辅助函数（不返回 Plugin）
senza.strategy.webhook_stream(buffer: int) -> tuple[WebhookChannel, EventStream]
senza.strategy.context_aware_compaction_prompt() -> tuple[str, str]
```

| 函数 | 说明 |
|------|------|
| `senza.strategy.safety_defaults()` | Bash 黑名单 + 路径穿越防护 |
| `senza.strategy.loop_safety(config=None)` | 死循环/重复/连续失败断路器 |
| `senza.strategy.status_panel()` | 状态栏 + `todo_write` 工具 |
| `senza.strategy.memory_defense()` | 持久记忆注入防御（默认文件集） |
| `MemoryDefensePluginBuilder` | 自定义文件集的记忆防御构建器 |
| `senza.strategy.injection_filter(patterns=None)` | 提示注入检测 |
| `senza.strategy.source_tag(entries)` | 外部内容 `<source>` 标签包裹 |
| `senza.strategy.vision_degrade()` | 视觉降级自愈（provider_error hook preset，issue #145） |
| `senza.strategy.observation_shielding(config=None)` | 隐藏旧 tool observation（transform_context hook preset） |
| `senza.strategy.project_instruction(env, config=None)` | 自动注入 CLAUDE.md 等项目指令 |
| `senza.strategy.audit(sink_path, trace_id=None, task_id=None)` | 工具调用审计日志（JSONL） |
| `senza.strategy.notify()` | LLM 主动通知用户 |
| `senza.strategy.tool_output_guard(env, config=None)` | 工具输出截断安全网 |
| `senza.strategy.webhook_stream(buffer)` | 外部事件触发流 |
| `senza.strategy.context_aware_compaction_prompt()` | 上下文感知 compaction 提示对 |

## Knowledge & Memory

```python
# 本地知识源（RAG）
senza.knowledge.local_source(
    path: str,
    source_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    domains: Optional[list[str]] = None,
    max_document_bytes: int = 1048576,
) -> KnowledgeSource

senza.knowledge.plugin(
    sources: list[KnowledgeSource],
    config: Optional[dict] = None,
) -> Plugin  # 注册 knowledge_search + knowledge_read 工具

# 记忆写入/删除（内置 store 是进程内演示实现）
senza.knowledge.memory_store(read_source_id: str) -> MemoryStore
senza.knowledge.secure_write_policy(config: Optional[dict] = None) -> MemoryWritePolicy
senza.knowledge.allow_all_gate() -> MemoryMutationGate
senza.knowledge.memory_plugin(
    source: KnowledgeSource,
    store: MemoryStore,
    policy: MemoryWritePolicy,
    gate: Optional[MemoryMutationGate] = None,
) -> Plugin  # 注册 memory_write + memory_forget 工具

# 会话历史召回
senza.knowledge.in_memory_session_recall_index() -> SessionRecallIndex
senza.knowledge.sqlite_session_recall_index(path: str) -> SessionRecallIndex
senza.knowledge.in_memory_session_repo() -> SessionRepo
senza.knowledge.jsonl_session_repo(path: str) -> SessionRepo  # JSONL 持久化会话仓库
senza.knowledge.session_recall_knowledge_source(
    repo: SessionRepo, index: SessionRecallIndex,
) -> SessionRecallKnowledgeSource
senza.knowledge.history_recall_plugin(
    source: SessionRecallKnowledgeSource,
    config: Optional[dict] = None,
) -> Plugin
```

> **Memory 边界**：`memory_plugin(..., gate=None)` 中 gate 可选，缺省使用完全放行的 `AllowAllGate`；生产环境应显式提供审批门禁。`local_source.source_id` 必须与 `memory_store.read_source_id` 一致，但当前内置 store 只将字节保留在进程内，不持久化，也不会自动同步到 `local_source`。
>
> **Recall 边界**：Python 已暴露 repo/index/source/plugin 的装配工厂；当前未暴露 projector/索引写入入口，因此使用 `history_recall_plugin` 前需要确保索引已由其他途径填充。

| 函数 | 说明 |
|------|------|
| `senza.knowledge.local_source(path, source_id, ...)` | 本地文档知识源 |
| `senza.knowledge.plugin(sources, config=None)` | `knowledge_search` + `knowledge_read` 工具 |
| `senza.knowledge.memory_store(read_source_id)` | 可写的进程内演示 store（`Mutex<Vec>`，不持久化） |
| `senza.knowledge.secure_write_policy(config=None)` | 注入安全写策略 |
| `senza.knowledge.allow_all_gate()` | 完全放行写门控 |
| `senza.knowledge.memory_plugin(source, store, policy, gate=None)` | `memory_write` + `memory_forget` 工具；gate 缺省为 `AllowAllGate` |
| `senza.knowledge.in_memory_session_recall_index()` | 内存会话索引 |
| `senza.knowledge.sqlite_session_recall_index(path)` | SQLite 持久化会话索引 |
| `senza.knowledge.in_memory_session_repo()` | 内存会话仓库 |
| `senza.knowledge.jsonl_session_repo(path)` | JSONL 持久化会话仓库 |
| `senza.knowledge.session_recall_knowledge_source(repo, index)` | 会话召回知识源 |
| `senza.knowledge.history_recall_plugin(source, config=None)` | 从已填充的召回索引检索并注入历史会话上下文 |

## Infra（审计 / Trace / 沙箱）

```python
# JSONL 审计 sink（SHA-256 哈希链完整性）
senza.JsonlAuditSink  # 类: append(record), validate(path) -> int

# 内存 trace 导出器（测试用）
senza.InMemoryTraceExporter  # 类: exported_span_count() -> int

# 沙箱 config 键：fs_allowlist / fs_denylist / work_dir / max_memory_mb /
# max_cpus / max_disk_mb / timeout_seconds / max_processes
# （max_processes 仅 Linux bwrap 生效——cgroup v2 进程数限制；seatbelt 忽略）
senza.infra.seatbelt_sandbox(config: Optional[dict] = None) -> Sandbox  # macOS
senza.infra.bwrap_sandbox(config: Optional[dict] = None) -> Sandbox     # Linux
```

| 类/函数 | 说明 |
|----------|------|
| `JsonlAuditSink` | JSONL 文件审计 sink，SHA-256 哈希链完整性 |
| `InMemoryTraceExporter` | 内存 trace 导出器，测试用 |
| `senza.infra.seatbelt_sandbox(config=None)` | macOS Seatbelt 沙箱 |
| `senza.infra.bwrap_sandbox(config=None)` | Linux Bubblewrap 沙箱 |

## CompositeJudge

```python
cj = senza.create_composite_judge()
cj.on("step_id", callback)    # 为指定 step 注册独立 judge
cj.fallback(callback)         # 注册兜底 judge（无匹配 on 时调用）
```

CompositeJudge 允许为不同 step 注册独立路由逻辑，避免在单个 judge 函数中写 if-else 链。
传入 `WorkflowEngine(workflow, provider, model, cj)` 即可使用。

## ResponseFormat

```python
senza.create_json_object_format()  # JSON object mode
senza.create_json_schema_format(
    name: str, schema: dict, strict: Optional[bool] = None,
) -> ResponseFormat
```

通过 `.response_format(fmt)` 注册到 HarnessBuilder，让 LLM 输出结构化 JSON。

## UsageLedger

```python
ledger = senza.UsageLedger()
harness = senza.HarnessBuilder("gpt-4o").provider("*", provider).usage_ledger(ledger).build()

snapshot = ledger.snapshot()  # dict: total cost, by_model, by_provider
```

多个 harness 共享同一 ledger 时，`snapshot()` 返回聚合成本。`harness.usage_ledger()` 返回当前快照（dict）。

## Event Streams（定时器 / 心跳 / Shell 监控）

```python
# 定时器：每隔 interval_ms 触发一次事件
timer_tool = senza.create_timer_stream(interval_ms=10000)

# 心跳：调用 handle.tick() 重置看门狗，超时触发事件
handle, heartbeat_tool = senza.create_heartbeat_stream(timeout_ms=30000)

# Shell 监控：启动子进程，超时或结束后触发事件
handle, shell_tool = senza.create_shell_monitor_stream(
    command, timeout_ms=30000, cwd=None,
)
```

返回的 tool 是 `WaitForExternalEventTool`，通过 `.with_external_tool(tool)` 注册到 WorkflowEngine。
LLM 调用对应的 wait 工具时暂停，直到事件触发。

## MCP（Model Context Protocol）

```python
# stdio 方式
config = senza.McpServerConfig.stdio(
    command: str, args: list[str] = ..., env: dict = ...,
)

# HTTP 方式
config = senza.McpServerConfig.http(
    url: str, headers: dict = ...,
)

# SSE 方式
config = senza.McpServerConfig.sse(
    url: str, headers: dict = ...,
)

# 注册到 builder
harness = (
    senza.HarnessBuilder("gpt-4o")
    .provider("*", provider)
    .mcp_server("db", config)
    .build()
)

# 或从配置文件加载
harness = (
    senza.HarnessBuilder("gpt-4o")
    .provider("*", provider)
    .mcp_config_file("mcp_servers.json")
    .build()
)

# 或注入预配置的 manager
manager = senza.McpManager()
manager.add_server("db", config)
harness = (
    senza.HarnessBuilder("gpt-4o")
    .provider("*", provider)
    .with_mcp_manager(manager)
    .build()
)
```

| 类/函数 | 说明 |
|----------|------|
| `McpServerConfig.stdio(command, args, env)` | stdio 传输 |
| `McpServerConfig.http(url, headers)` | HTTP 传输 |
| `McpServerConfig.sse(url, headers)` | SSE 传输 |
| `McpManager` | 多 server 生命周期管理器 |
| `.mcp_server(name, config)` | 注册单个 MCP server |
| `.mcp_config_file(path)` | 从 JSON 文件批量加载 |
| `.with_mcp_manager(manager)` | 注入预配置 manager |

## Session Viewer

```bash
# CLI
python -m senza.viewer /path/to/sessions [--port PORT]
```

```python
# 编程式
import senza.viewer

senza.viewer.serve("/path/to/sessions")  # 阻塞，自动打开浏览器
```

支持：session 列表、分支树切换、消息渲染（user/assistant/tool-result）、thinking 和 tool-use 折叠、token 用量统计、config entry（model change / compaction / label 等）、图片内联。
