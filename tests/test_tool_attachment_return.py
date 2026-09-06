"""工具返回 Attachment（多模态结果）+ report_duration 透传。"""

import base64
import json

import pytest
import senza

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _make_tool(callback, **kwargs):
    return senza.create_tool(
        "test_tool",
        "Test tool",
        {"type": "object", "properties": {}},
        callback,
        **kwargs,
    )


# ── 工具返回 Attachment ──────────────────────────────────────────────


def test_bare_attachment_return():
    """裸 Attachment 返回 → 单个 image 块。"""
    tool = _make_tool(lambda args, ctx: senza.image_base64(PNG_1X1))
    result = tool.drive({})
    assert len(result["content"]) == 1
    block = result["content"][0]
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"


def test_attachment_document_return():
    tool = _make_tool(
        lambda args, ctx: senza.document_url("https://example.com/d.pdf", name="d.pdf")
    )
    result = tool.drive({})
    block = result["content"][0]
    assert block["type"] == "document"
    assert block["media_type"] == "application/pdf"
    assert block["data"]["type"] == "url"


def test_list_with_attachment_and_str():
    """list [Attachment, str] → [image 块, text 块]。"""
    tool = _make_tool(lambda args, ctx: [senza.image_base64(PNG_1X1), "caption"])
    result = tool.drive({})
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "image"
    assert result["content"][1] == {"type": "text", "text": "caption"}


def test_content_list_with_attachment():
    """dict content 列表里混 Attachment。"""
    tool = _make_tool(
        lambda args, ctx: {
            "content": [senza.image_base64(PNG_1X1), {"type": "text", "text": "see above"}],
            "terminate": False,
        }
    )
    result = tool.drive({})
    assert len(result["content"]) == 2
    assert result["content"][1]["text"] == "see above"


def test_str_return_regression():
    """回归：str 返回路径不受影响。"""
    tool = _make_tool(lambda args, ctx: "hello")
    result = tool.drive({})
    assert result["content"][0]["text"] == "hello"


def test_dict_return_regression():
    """回归：无 content 的 dict 返回路径不受影响。"""
    tool = _make_tool(lambda args, ctx: {"status": "ok"})
    result = tool.drive({})
    parsed = json.loads(result["content"][0]["text"])
    assert parsed == {"status": "ok"}


def test_list_with_non_attachment_non_str_rejects():
    """list 里非 Attachment 非 str → ValueError。"""
    tool = _make_tool(lambda args, ctx: [123])
    with pytest.raises(Exception):
        tool.drive({})


# ── report_duration ──────────────────────────────────────────────────


def test_create_tool_accepts_report_duration():
    tool = senza.create_tool(
        "slow",
        "slow tool",
        {"type": "object", "properties": {}},
        lambda args, ctx: "done",
        report_duration=True,
    )
    assert tool.name == "slow"


def test_report_duration_default_off():
    """不带 report_duration 时默认关闭（签名兼容）。"""
    tool = _make_tool(lambda args, ctx: "ok")
    assert tool.name == "test_tool"


# ── dict content 块的 image/document 支持（issue #145）───────────────


def test_dict_qevos_style_image_block():
    """Qevos 风格 {"type":"image","media_type","data"} → image 块。"""
    tool = _make_tool(
        lambda args, ctx: {
            "content": [
                {"type": "text", "text": "screenshot"},
                {"type": "image", "media_type": "image/png", "data": base64.b64encode(PNG_1X1).decode()},
            ]
        }
    )
    result = tool.drive({})
    assert len(result["content"]) == 2
    block = result["content"][1]
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    assert base64.b64decode(block["source"]["data"]) == PNG_1X1


def test_dict_image_block_defaults_png():
    """缺 media_type → image/png。"""
    tool = _make_tool(
        lambda args, ctx: {"content": [{"type": "image", "data": "AAAA"}]}
    )
    result = tool.drive({})
    block = result["content"][0]
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/png"


def test_dict_image_url_block():
    """{"type":"image","url":...} → url image 块。"""
    tool = _make_tool(
        lambda args, ctx: {"content": [{"type": "image", "url": "https://example.com/x.png"}]}
    )
    result = tool.drive({})
    block = result["content"][0]
    assert block["type"] == "image"
    assert block["source"]["type"] == "url"
    assert block["source"]["url"] == "https://example.com/x.png"


def test_dict_image_url_shorthand_type():
    """{"type":"image_url","url":...} 构造器风格。"""
    tool = _make_tool(
        lambda args, ctx: {"content": [{"type": "image_url", "url": "https://example.com/x.png"}]}
    )
    result = tool.drive({})
    assert result["content"][0]["source"]["type"] == "url"


def test_dict_document_base64_block():
    """{"type":"document_base64",...} → document 块。"""
    tool = _make_tool(
        lambda args, ctx: {
            "content": [
                {
                    "type": "document_base64",
                    "name": "d.pdf",
                    "media_type": "application/pdf",
                    "data": "AAAA",
                }
            ]
        }
    )
    result = tool.drive({})
    block = result["content"][0]
    assert block["type"] == "document"
    assert block["name"] == "d.pdf"
    assert block["data"]["type"] == "base64"


def test_dict_unsupported_type_still_errors_with_hint():
    """未知类型仍报错，且提示 Attachment 迁移路径。"""
    tool = _make_tool(lambda args, ctx: {"content": [{"type": "audio", "data": "x"}]})
    with pytest.raises(RuntimeError, match="Attachment"):
        tool.drive({})


def test_dict_image_block_missing_data_and_url():
    """image 块缺 data 和 url → 明确报错。"""
    tool = _make_tool(lambda args, ctx: {"content": [{"type": "image"}]})
    with pytest.raises(RuntimeError, match="data.*or.*url"):
        tool.drive({})
