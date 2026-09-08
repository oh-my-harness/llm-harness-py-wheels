"""Multimodal live tests: image/document attachments through real LLM endpoints.

No key → all real tests skip (provider_or_skip). Offline smoke test always runs.
Document tests auto-skip on endpoints that reject document input.
"""

import base64

import pytest
import senza
from base import (
    assert_no_error,
    assert_settled,
    document_provider_or_skip,
    make_harness,
    provider_or_skip,
    run_prompt,
    text_of,
)

# 64x64 solid red PNG, generated programmatically so no network fetch is needed.
PNG_64X64_RED_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAlElEQVR4nO3QMREAMBDDsPAn/YW"
    "hoR60+7zb7mfTAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAV"
    "oDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHaA3QAVoDdIDWAB2gNUAHa"
    "A3QAVoDdIDWAB2gNUAHaA9DiOHSbdjxEgAAAABJRU5ErkJggg=="
)


def _red_png_bytes() -> bytes:
    return base64.b64decode(PNG_64X64_RED_B64)


def _vision_provider():
    """(provider, model) for a vision-capable endpoint.

    SENZA_VISION_MODEL overrides; default GLM-5.3-Flash (DeepSeek-V4-Flash
    rejects image parts with a 400, so vision tests cannot run on it).
    """
    import os

    provider = provider_or_skip()
    model = os.environ.get(
        "SENZA_VISION_MODEL", os.environ.get("SENZA_LIVE_MODEL", "GLM-5.3-Flash")
    )
    return provider, model


def test_multimodal_constructs_offline():
    """No key needed: validates constructor + prompt API signatures."""
    stub = senza.providers.openai(api_key="sk-test")
    h = make_harness(stub, lambda b: b.max_tokens(50))
    a = senza.image_base64(_red_png_bytes())
    assert a is not None
    assert senza.image_url("https://example.com/x.png") is not None
    assert senza.document_url("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf") is not None
    # signatures accept attachments kwarg (no real call needed for steer on idle harness)
    h.steer("ignored", attachments=[a])
    h.follow_up("ignored", attachments=None)
    h.next_turn("ignored", attachments=[senza.image_url("https://example.com/x.png")])


def test_prompt_image_base64():
    provider, model = _vision_provider()
    h = make_harness(provider, lambda b: b.max_tokens(300), model=model)
    ev = run_prompt(
        h,
        "What color is this image? Answer with one word.",
        attachments=[senza.image_base64(_red_png_bytes())],
    )
    assert_settled(ev)
    assert_no_error(ev)
    reply = text_of(ev).strip().lower()
    assert reply, "expected non-empty reply"
    assert "red" in reply, f"expected the model to see red, got: {reply}"


def test_prompt_image_url():
    """URL 图片。断言模型给出非空描述（内容不可预测，不断言具体颜色）。"""
    provider, model = _vision_provider()
    h = make_harness(provider, lambda b: b.max_tokens(300), model=model)
    ev = run_prompt(
        h,
        "Describe this image in one sentence.",
        attachments=[senza.image_url("https://picsum.photos/200")],
    )
    assert_settled(ev)
    assert_no_error(ev)
    reply = text_of(ev).strip()
    assert reply, "expected non-empty reply"
    assert len(reply) > 10, f"expected a substantive description, got: {reply!r}"


def test_tool_returns_image():
    """Python 工具返回 Attachment → 模型能看见图片内容（工具结果多模态链路）。"""

    def image_tool():
        return senza.create_tool(
            name="get_screenshot",
            description="Returns a screenshot image. Call it to see the image.",
            parameters_schema='{"type":"object","properties":{}}',
            callback=lambda args, ctx: senza.image_base64(_red_png_bytes()),
        )

    provider, model = _vision_provider()
    h = make_harness(
        provider,
        lambda b: b.max_tokens(300).tool(image_tool()),
        model=model,
    )
    ev = run_prompt(h, "Call get_screenshot and tell me what color the image is. One word.")
    assert_settled(ev)
    assert_no_error(ev)
    reply = text_of(ev).lower()
    assert reply.strip(), "expected non-empty reply"
    assert "red" in reply, f"expected the model to see red via tool, got: {reply}"


def _is_capability_rejection(e: Exception) -> bool:
    msg = str(e).lower()
    return any(
        k in msg
        for k in ("document", "unsupported", "not support", "file_url", "invalid request", "bad request", "does not support")
    )


def test_document_url():
    """PDF URL 附件，走已知支持文档的端点（base.py 自动发现，无则 skip）。

    实测（2026-08-29）：hyper-op GLM 拒绝文档输入 → 那里 skip；
    .env 网关 claude-sonnet-4-6 完整读出 PDF → 断言内容。
    """
    provider, model = document_provider_or_skip()
    h = make_harness(provider, lambda b: b.max_tokens(300), model=model)
    try:
        ev = run_prompt(
            h,
            "What is this document about? One sentence.",
            attachments=[
                senza.document_url(
                    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                    name="dummy.pdf",
                )
            ],
        )
    except Exception as e:  # noqa: BLE001 — provider capability rejection
        if _is_capability_rejection(e):
            pytest.skip(f"endpoint does not support document input: {e}")
        raise
    assert_settled(ev)
    assert_no_error(ev)
    reply = text_of(ev).strip()
    assert reply, "expected non-empty reply"
    assert "pdf" in reply.lower() or "document" in reply.lower() or "dummy" in reply.lower(), (
        f"expected the model to read the PDF, got: {reply!r}"
    )


def test_get_messages_persists_image():
    provider, model = _vision_provider()
    h = make_harness(provider, lambda b: b.max_tokens(200), model=model)
    run_prompt(h, "Briefly: what color is this image?", attachments=[senza.image_base64(_red_png_bytes())])
    msgs = h.get_messages()
    user_blocks = [b for m in msgs if m.get("role") == "user" for b in m.get("content", [])]
    assert any(b.get("type") == "image" for b in user_blocks), (
        f"expected image block in session, got: {user_blocks}"
    )
