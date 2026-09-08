//! Rust 集成测试：在 tokio runtime 内用 GIL 调用 PyTool。
//!
//! 需要 Python 解释器，用 `cargo test --test pytool_integration -- --ignored` 运行。
//! 前置：`maturin develop` 或设置 PYTHONPATH。

use std::sync::Arc;

use llm_harness_loop::test_utils::{NoOpEnv, test_assistant_message};
use llm_harness_types::{DataBlock, RunContext, RunRequest, ToolContext};
use tokio_util::sync::CancellationToken;

#[tokio::test]
#[ignore = "requires Python interpreter — run with --ignored"]
async fn pytool_executes_sync_callback() {
    // Python::attach 会按需初始化解释器（PyO3 0.29 移除了 prepare_freethreaded_python）。
    let env = Arc::new(NoOpEnv);

    // 在 GIL 内创建 PyTool
    let tool = pyo3::Python::attach(|py| {
        let callback = py
            .eval(
                c"lambda args, ctx: {\"content\": [{\"type\": \"text\", \"text\": args[\"text\"]}], \"terminate\": False}",
                None,
                None,
            )
            .unwrap()
            .unbind();
        let schema =
            serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}});
        Arc::new(senza::core::pytool::PyTool::new(
            "echo".into(),
            "echo".into(),
            schema,
            callback,
            false,
        )) as Arc<dyn llm_harness_types::Tool>
    });

    // 验证 tool.execute 可在 async 上下文调用
    let ctx = ToolContext {
        run: Arc::new(RunContext::new(RunRequest::default())),
        env: env.clone(),
        abort: CancellationToken::new(),
        tool_use_id: "test-1".into(),
        turn_index: 0,
        assistant_message: Arc::new(test_assistant_message(vec![])),
        update_tx: tokio::sync::mpsc::channel(1).0,
    };
    let result = tool
        .execute(serde_json::json!({"text": "hello"}), &ctx)
        .await
        .unwrap();
    assert_eq!(result.model_content.len(), 1);
    match &result.model_content[0] {
        DataBlock::Text { text, .. } => assert_eq!(text, "hello"),
        _ => panic!("expected text block"),
    }
}
