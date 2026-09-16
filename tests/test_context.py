from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from hippo.agent.context import compress_messages, count_tokens


def _convo(n_turns: int, chunk: str = "x" * 400):
    msgs = [SystemMessage(content="sys"), HumanMessage(content="task")]
    for i in range(n_turns):
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "read", "args": {"path": f"f{i}"}, "id": f"c{i}"}],
            )
        )
        msgs.append(ToolMessage(content=f"{i}:{chunk}", tool_call_id=f"c{i}"))
    return msgs


def test_count_tokens_monotonic() -> None:
    assert count_tokens(_convo(1)) < count_tokens(_convo(3))


def test_no_compression_under_budget() -> None:
    msgs = _convo(2)
    out, info = compress_messages(msgs, summarize=lambda _: "S", token_budget=10**6)
    assert out is msgs and info is None


def test_compression_keeps_head_and_tail_and_summarizes_middle() -> None:
    msgs = _convo(6)
    seen: list[str] = []

    def summarize(body: str) -> str:
        seen.append(body)
        return "SUMMARY"

    out, info = compress_messages(msgs, summarize=summarize, token_budget=200, keep_tail=4)
    assert info and info["dropped_messages"] > 0
    assert out[0] is msgs[0] and out[1] is msgs[1]
    assert isinstance(out[2], SystemMessage) and "SUMMARY" in out[2].content
    assert out[-4:] == msgs[-4:]  # last two tool turns verbatim
    assert "f0" in seen[0] and "f5" not in seen[0]  # middle summarized, tail untouched
    assert info["after_tokens"] < info["before_tokens"]


def test_tail_never_splits_tool_call_from_result() -> None:
    msgs = _convo(6)
    out, _ = compress_messages(msgs, summarize=lambda _: "S", token_budget=200, keep_tail=3)
    # first message after the summary must be an AI call, not an orphan ToolMessage
    assert isinstance(out[3], AIMessage)


def test_second_compression_folds_previous_note() -> None:
    msgs = _convo(6)
    out, _ = compress_messages(msgs, summarize=lambda _: "FIRST", token_budget=200)
    out += _convo(4)[2:]  # more turns after the note
    bodies: list[str] = []
    out2, info = compress_messages(
        out, summarize=lambda b: bodies.append(b) or "SECOND", token_budget=200
    )
    assert info
    notes = [m for m in out2 if isinstance(m, SystemMessage) and "Compressed" in m.content]
    assert len(notes) == 1 and "SECOND" in notes[0].content
    assert "FIRST" in bodies[0]
