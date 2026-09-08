//! Sub-agent spawn infrastructure — Python-facing `enable_spawn()` wiring.
//!
//! Mirrors the runtime's spawn assembly:
//! 1. `MessageBus` via `MessageBus::new()`.
//! 2. `HarnessSubAgentSpawner` (model, client, session_dir, bus, NoopPlugin).
//! 3. `SpawnPlugin` via `SpawnPlugin::new(bus)`.
//! 4. Five tools: `SpawnAgentTool`, `MessageSubagentTool`, `AwaitSubagentReplyTool`,
//!    `QuerySubagentTool`, `AbortSubagentTool`.
//! 5. Builder: tools + `install(plugin)` + `convert_to_llm(...)`.
//! 6. Post-build: `plugin.set_harness_weak(weak)`.

use std::path::Path;
use std::sync::Arc;

use llm_harness_agent::HarnessBuilder;
use llm_harness_agent::{AgentHarness, Plugin};
use llm_harness_loop::convert::DefaultConvertToLlm;
use llm_harness_sandbox::os::OsEnvFactory;
use llm_harness_subagents::delivery::SubAgentMessageConverter;
use llm_harness_subagents::message_bus::{MAIN_AGENT_ID, MessageBus};
use llm_harness_subagents::plugin::SpawnPlugin;
use llm_harness_subagents::spawner::{HarnessSubAgentSpawner, JsonlSessionFactory};
use llm_harness_subagents::tools::{
    AbortSubagentTool, AwaitSubagentReplyTool, MessageSubagentTool, QuerySubagentTool,
    SpawnAgentTool,
};
use llm_harness_types::Tool;

/// Post-build spawn wiring state. Held across `build()` and applied
/// to the constructed `AgentHarness`.
pub(crate) struct SpawnWiring {
    plugin: Arc<SpawnPlugin>,
}

impl SpawnWiring {
    /// Apply post-build: link the SpawnPlugin to the harness via weak ref.
    /// Must be called after `build()` returns the harness.
    pub(crate) fn post_build(&self, harness: &Arc<AgentHarness>) {
        self.plugin.set_harness_weak(Arc::downgrade(harness));
    }
}

/// A no-op plugin for sub-agents — prevents recursive spawning.
struct NoopPlugin;

impl Plugin for NoopPlugin {
    fn name(&self) -> &str {
        "senza-noop-spawn"
    }
}

/// Wire spawn infrastructure into the builder and return post-build wiring state.
///
/// - Adds `SpawnAgentTool`, `MessageSubagentTool`, `AwaitSubagentReplyTool`,
///   `QuerySubagentTool`, `AbortSubagentTool` to the builder.
/// - Installs `SpawnPlugin` (registers before_run, after_turn, after_run,
///   on_abort hooks that manage sub-agent event delivery and idle wakeup).
/// - Sets `convert_to_llm(DefaultConvertToLlm + SubAgentMessageConverter)`.
/// - Returns `(modified_builder, SpawnWiring)` for post-build hook application.
pub(crate) fn wire_spawn(
    mut builder: HarnessBuilder,
    cfg: crate::core::pybuilder::SpawnConfig,
) -> (HarnessBuilder, Option<SpawnWiring>) {
    // 1. Message bus (Arc<MessageBus>, no event channel).
    let bus = MessageBus::new();

    // 2. Spawner.
    let spawner = HarnessSubAgentSpawner::new(
        cfg.model,
        cfg.client,
        cfg.session_dir,
        bus.clone(),
        |_cwd: &Path, _bus, _agent_id: &str| Box::new(NoopPlugin) as Box<dyn Plugin>,
    )
    .env_factory(Arc::new(OsEnvFactory))
    .session_factory(Arc::new(JsonlSessionFactory))
    .max_concurrent(cfg.max_concurrent);
    let spawner = Arc::new(spawner);

    // 3. SpawnPlugin — replaces AsyncSpawnHook + IdleWatcher + AbortCascadeHook.
    let plugin = SpawnPlugin::new(bus.clone());

    // 4. Register tools + install plugin + convert_to_llm.
    builder = builder
        .tool(Arc::new(SpawnAgentTool::new(spawner.clone())) as Arc<dyn Tool>)
        .tool(Arc::new(MessageSubagentTool::new(bus.clone(), MAIN_AGENT_ID)) as Arc<dyn Tool>)
        .tool(Arc::new(AwaitSubagentReplyTool::new(bus.clone(), MAIN_AGENT_ID)) as Arc<dyn Tool>)
        .tool(Arc::new(QuerySubagentTool::new(bus.clone())) as Arc<dyn Tool>)
        .tool(Arc::new(AbortSubagentTool::new(bus.clone())) as Arc<dyn Tool>)
        .install(plugin.as_ref())
        .convert_to_llm(Some(Arc::new(
            DefaultConvertToLlm::new().with_custom_converter(Arc::new(SubAgentMessageConverter)),
        )));

    (builder, Some(SpawnWiring { plugin }))
}
