"""41 — Human-in-the-Loop: pause for external events and human approval.

Mirrors runtime `06_human_in_the_loop.py`. Demonstrates:
  - create_event_channel() for external event injection
  - The LLM calls wait_for_external_event to pause for human input
  - Submit events from another thread
  - create_human_approval_channel() with timeout + fail-safe default

By default a background thread simulates the reviewer after 3 seconds
(unattended run). Pass --interactive to BE the reviewer yourself in the
terminal: the script prompts you when the agent asks for input.

Run:
  source ~/.omp_llm_env && python live-tests/examples/41_human_in_the_loop.py
  # interactive (you are the human):
  source ~/.omp_llm_env && python live-tests/examples/41_human_in_the_loop.py --interactive
"""

import sys
import threading
import time

import senza
from _common import live_model, require_provider


def main(interactive: bool = False) -> None:
    provider = require_provider()
    finished = threading.Event()

    # Create an event channel — the wait_for_external_event tool will be
    # available to the LLM. When it calls this tool, execution pauses until
    # handle.submit() is called from another thread.
    handle, wait_tool = senza.create_event_channel("review-task")

    workflow = {
        "entry_step": "draft",
        "steps": [
            {
                "id": "draft",
                "name": "Draft",
                "prompt": "Draft a short email to a client about a project delay. "
                "Then call wait_for_external_event to get approval.",
                "allowed_tools": ["wait_for_external_event"],
            },
        ],
        "edges": [],
    }

    engine = senza.WorkflowEngine(
        workflow, provider, live_model(), senza.create_judge(lambda ctx: "done")
    ).with_external_tool(wait_tool)

    def auto_approve_hook():
        # 无人值守模式：tool 被调用的瞬间 submit——确定性时序，
        # 不用 sleep 赌 LLM 何时发起请求。
        def on_tool_call(ctx: dict) -> None:
            if ctx.get("tool_name") != "wait_for_external_event" or finished.is_set():
                return "allow"
            print("\n[Human reviewer: approving...]")
            handle.submit("approved", {"feedback": "Looks good, send it!"})
            return "allow"

        return senza.hooks.before_tool_call(on_tool_call)

    def terminal_review_hook():
        # 真实人机交互：before_tool_call hook 在 LLM 调用
        # wait_for_external_event 的瞬间触发——提示一次、读一行输入。
        # 被拒绝后 LLM 会修订草稿再次调用 tool，自然再次提示，
        # 不会空转轮询，也不会提前排队事件。
        def on_tool_call(ctx: dict) -> None:
            if ctx.get("tool_name") != "wait_for_external_event" or finished.is_set():
                return "allow"
            try:
                answer = input("\n[Reviewer] approve the draft? (y/n): ").strip().lower()
            except EOFError:
                # 无 stdin（管道 EOF）：不 submit，workflow 由 tool 超时收尾
                print("\n[Reviewer] no stdin — leaving the decision to timeout")
                finished.set()
                return
            decision = "approved" if answer.startswith("y") else "rejected"
            handle.submit(decision, {"feedback": f"Reviewer says {decision}"})
            return "allow"

        return senza.hooks.before_tool_call(on_tool_call)

    engine = engine.with_hooks([terminal_review_hook() if interactive else auto_approve_hook()])

    print("Running workflow with human-in-the-loop...")
    engine.run()
    finished.set()

    print(f"\nFinal state: {engine.state()}")
    history = engine.step_history()
    for record in history:
        result = record.get("result") or {}
        output = (result.get("output") or "")[:120]
        if output:
            print(f"  {record['step_id']}: {output}")


def demo_approval_channel(interactive: bool = False) -> None:
    """Human approval channel with timeout + fail-safe default.

    Same pause-and-resume pattern as create_event_channel, but with
    approve/deny semantics: the tool returns a structured decision and
    applies the fail-safe default ("deny") on timeout.
    """
    provider = require_provider()
    finished = threading.Event()

    handle, approval_tool = senza.create_human_approval_channel(
        "deploy-gate", timeout_seconds=120.0, default="deny"
    )

    workflow = {
        "entry_step": "deploy",
        "steps": [
            {
                "id": "deploy",
                "name": "Deploy",
                "prompt": "You are about to deploy to production. "
                "Call request_human_approval to ask for permission, "
                "then summarize the decision in one sentence.",
                "allowed_tools": ["request_human_approval"],
            },
        ],
        "edges": [],
    }

    engine = senza.WorkflowEngine(
        workflow, provider, live_model(), senza.create_judge(lambda ctx: "done")
    ).with_external_tool(approval_tool)

    def auto_approve_hook():
        # 无人值守模式：before_tool_call 触发即知 LLM 已发起审批请求，
        # 由后台线程 submit——human channel 的 pending request_id 在
        # wrapper.execute 里才记录（晚于 hook），重试直到 submit 成功。
        def on_tool_call(ctx: dict) -> None:
            if ctx.get("tool_name") != "request_human_approval" or finished.is_set():
                return "allow"
            print("\n[Human reviewer: approving...]")
            # The handle auto-injects the pending request_id; the caller
            # only supplies the decision.
            threading.Thread(target=_submit_with_retry, args=("approve",), daemon=True).start()
            return "allow"

        return senza.hooks.before_tool_call(on_tool_call)

    def _submit_with_retry(decision: str):
        while not finished.is_set():
            try:
                handle.submit(decision, {"decision": decision})
                return
            except RuntimeError:
                # tool 尚未记录 pending request_id（wrapper.execute 晚于
                # before_tool_call hook），稍后重试。
                time.sleep(0.05)

    def terminal_review_hook():
        # 真实人机交互：before_tool_call hook 在 LLM 调用
        # request_human_approval 的瞬间触发。input() 在 hook 线程阻塞
        # 即是"暂停等人"；submit 由后台线程带重试发出——pending
        # request_id 在 wrapper.execute 里才记录（晚于 hook 返回）。
        def on_tool_call(ctx: dict) -> None:
            if ctx.get("tool_name") != "request_human_approval" or finished.is_set():
                return "allow"
            try:
                answer = (
                    input("\n[Reviewer] allow deployment to production? (y/n): ").strip().lower()
                )
            except EOFError:
                # 无 stdin（管道 EOF）：不 submit，workflow 由超时默认值收尾
                print("\n[Reviewer] no stdin — leaving the decision to timeout")
                finished.set()
                return
            decision = "approve" if answer.startswith("y") else "deny"
            threading.Thread(target=_submit_with_retry, args=(decision,), daemon=True).start()
            return "allow"

        return senza.hooks.before_tool_call(on_tool_call)

    engine = engine.with_hooks([terminal_review_hook() if interactive else auto_approve_hook()])

    print("\nRunning workflow with human approval gate...")
    engine.run()
    finished.set()


if __name__ == "__main__":
    interactive = "--interactive" in sys.argv
    main(interactive)
    demo_approval_channel(interactive)
