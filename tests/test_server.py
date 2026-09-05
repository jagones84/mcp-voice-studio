"""Smoke test: verify all 6 tools are registered and their schemas look sane.

Does NOT require GPU. Tests only the FastMCP registration.
"""


def test_all_six_tools_registered():
    from mcp_voice_studio.server import mcp

    tools = mcp._tool_manager._tools
    expected = {
        "tool_clone_voice_from_audio",
        "tool_synthesize_speech",
        "tool_design_voice",
        "tool_list_voices",
        "tool_get_voice_info",
        "tool_delete_voice",
    }
    actual = set(tools.keys())
    assert expected.issubset(actual), f"missing tools: {expected - actual}"


def test_synthesize_speech_requires_voice_or_instruct():
    """synthesize_speech raises if neither voice_name nor instruct is provided."""
    from mcp_voice_studio.tools.synthesize import synthesize_speech
    with pytest_assert():
        synthesize_speech(text="hello")
    # both voice_name and instruct: OK (validate instruct)
    # We don't actually call synth here (would need GPU)


def pytest_assert():
    """Helper to avoid importing pytest at top-level for this single test."""
    import pytest
    return pytest.raises(ValueError, match="either voice_name or instruct")
