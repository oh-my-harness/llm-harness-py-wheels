//! Rust 集成测试：验证 async Python tool callback 不死锁。
//!
//! 需要 Python 解释器，用 `cargo test --test async_tool_integration -- --ignored` 运行。
//! 前置：`maturin develop` 或设置 PYTHONPATH。

use llm_harness_loop::test_utils::{NoOpEnv, test_assistant_message};
use pyo3::types::PyDictMethods;
use std::sync::Arc;

use llm_harness_types::{DataBlock, RunContext, RunRequest, ToolContext};
use tokio_util::sync::CancellationToken;

/// 验证 async Python callback 在 tokio runtime 内执行不死锁。
/// 如果死锁，10 秒超时会触发 panic。
#[tokio::test]
#[ignore = "requires Python interpreter — run with --ignored"]
async fn async_pytool_executes_without_deadlock() {
    let env = Arc::new(NoOpEnv);

    // 在 GIL 内创建 PyTool，callback 为 async def
    let tool = pyo3::Python::attach(|py| {
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            c"async def cb(args, ctx):\n    import asyncio\n    await asyncio.sleep(0.01)\n    return {\"content\": [{\"type\": \"text\", \"text\": args[\"text\"]}], \"terminate\": False}\n",
            None,
            Some(&locals),
        )
        .unwrap();
        let callback = locals.get_item("cb").unwrap().unwrap().unbind();
        let schema =
            serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}});
        Arc::new(senza::core::pytool::PyTool::new(
            "async_echo".into(),
            "async echo".into(),
            schema,
            callback,
            false,
        )) as Arc<dyn llm_harness_types::Tool>
    });

    let ctx = ToolContext {
        run: Arc::new(RunContext::new(RunRequest::default())),
        env: env.clone(),
        abort: CancellationToken::new(),
        tool_use_id: "test-async-1".into(),
        turn_index: 0,
        assistant_message: Arc::new(test_assistant_message(vec![])),
        update_tx: tokio::sync::mpsc::channel(1).0,
    };

    // 10 秒超时——如果死锁会触发 panic
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tool.execute(serde_json::json!({"text": "async hello"}), &ctx),
    )
    .await
    .expect("async tool execution timed out — likely deadlock")
    .unwrap();

    assert_eq!(result.model_content.len(), 1);
    match &result.model_content[0] {
        DataBlock::Text { text, .. } => assert_eq!(text, "async hello"),
        _ => panic!("expected text block"),
    }
}
