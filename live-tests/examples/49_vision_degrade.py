"""49 — Vision Degrade: auto-recover from text-only provider rejection.

Demonstrates:
  - Building a multimodal history (image attachment) with the live model
  - Attaching senza.strategy.vision_degrade() via builder.provider_error_hook()
  - Following up on the multimodal history: if the provider rejects the
    request that carries the image block (HTTP 400 InvalidRequest), the hook
    repairs the history by stripping vision blocks and retries the same turn
  - Without the hook, this run dies with the provider error (issue #145 gap 2)

Note: the recovery path needs a live endpoint that actually rejects image
blocks for the configured model. With endpoints that silently accept images,
this example still shows the hook mounting contract (a settled run, no error).

Run:
  source ~/.omp_llm_env && python live-tests/examples/49_vision_degrade.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import senza
from _common import make_example_harness, require_provider, text_of
from base import run_prompt


def _red_png_bytes() -> bytes:
    """Minimal 64x64 red PNG (same fixture as live-tests/test_multimodal.py)."""
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAlElEQVR4nO3QMREAMBDDsPAn/YW"
        "hoR60+7zb7mfTAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QA"
        "VoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNU"
        "AHaA3QAVoDdIDWAB2gNUAHaA9DiOHSbdjxEgAAAABJRU5ErkJggg=="
    )


def main() -> None:
    require_provider()

    # The vision_degrade hook mounts on the builder's provider_error slot.
    # It fires only when the provider rejects the request itself
    # (InvalidRequest), strips image/document blocks from history, and
    # retries the same turn — at most once per run.
    harness = make_example_harness(
        lambda b: b.provider_error_hook(senza.strategy.vision_degrade()).max_tokens(300)
    )

    # 1. Seed a multimodal history: an image the model can see.
    events = run_prompt(
        harness,
        "Describe this image in one word, then say 'done'.",
        attachments=[senza.image_base64(_red_png_bytes())],
    )
    first_reply = text_of(events).strip()
    print(f"Vision-capable turn reply: {first_reply!r}")

    # 2. Follow up on the same (now multimodal) history. If the live model
    #    rejects image blocks, the hook strips them and retries in-place.
    harness.follow_up("In one sentence: what did the image show?")
    harness.continue_run()
    harness.wait_for_settled()
    follow_reply = (harness.last_response() or "").strip()
    print(f"Follow-up reply: {follow_reply!r}")
    print(
        "\nOK: run completed with the vision_degrade hook mounted. "
        "On endpoints that reject image blocks for text-only models, the hook "
        "strips vision blocks and retries instead of failing the run."
    )


if __name__ == "__main__":
    main()
