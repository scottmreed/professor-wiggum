"""ConversationThread: checkpoints, rewind, branching notes and compaction."""
from __future__ import annotations

from mechanistic_agent.core.conversation_thread import ConversationThread


def _thread_with_root() -> ConversationThread:
    thread = ConversationThread(max_prompt_chars=100_000)
    thread.set_root(
        [
            {"role": "user", "content": "pre-analysis prompt"},
            {"role": "assistant", "content": "Called unified_preanalysis_result with: {}"},
        ]
    )
    return thread


def _accept(thread: ConversationThread, step: int, *, rank: int = 1) -> None:
    thread.append_exchange(step, f"propose step {step}", f"answer step {step}", full=step == 1)
    thread.on_accept(
        step,
        rank=rank,
        summary=f"step {step}: A>>B{step}",
        next_note=f"Your rank {rank} candidate for step {step} was applied.",
    )


def test_root_is_checkpoint_zero_and_first_turn_is_full() -> None:
    thread = _thread_with_root()
    assert thread.checkpoints["accepted:0"] == 2
    assert thread.needs_full_prompt() is True
    _accept(thread, 1)
    assert thread.needs_full_prompt() is False
    assert [m["role"] for m in thread.messages()] == ["user", "assistant", "user", "assistant"]


def test_fork_drops_failed_proposal_and_keeps_validated_steps() -> None:
    thread = _thread_with_root()
    _accept(thread, 1)
    thread.append_exchange(2, "propose step 2", "bad answer", full=False)
    thread.add_avoid(2, "CC[O+] via SMIRKS x — failed: atom_balance")

    dropped = thread.fork_to_step(1)

    assert dropped == 2
    assert all("bad answer" not in m["content"] for m in thread.messages())
    preamble = thread.preamble(2)
    assert "atom_balance" in preamble
    assert "rank 1 candidate for step 1 was applied" in preamble


def test_fork_is_noop_right_after_accept() -> None:
    thread = _thread_with_root()
    _accept(thread, 1)
    assert thread.fork_to_step(1) == 0


def test_accept_clears_notes_for_that_step_and_sets_next_note() -> None:
    thread = _thread_with_root()
    _accept(thread, 1)
    thread.add_avoid(2, "x failed")
    _accept(thread, 2)
    assert "x failed" not in thread.preamble(3)
    assert "step 2 was applied" in thread.preamble(3)
    assert thread.preamble(2) == ""


def test_avoid_notes_are_deduplicated_capped_and_truncated() -> None:
    thread = _thread_with_root()
    for i in range(10):
        thread.add_avoid(1, f"note {i} " + "z" * 500)
    thread.add_avoid(1, "note 9 " + "z" * 500)
    notes = thread.avoid[1]
    assert len(notes) == 6
    assert all(len(note) <= 300 for note in notes)


def test_backtrack_fork_keeps_the_branch_exchange_and_drops_later_steps() -> None:
    thread = _thread_with_root()
    _accept(thread, 1)
    _accept(thread, 2)
    _accept(thread, 3)
    thread.add_avoid(4, "dead end")

    thread.fork_to_step(2)
    thread.on_accept(
        2,
        rank=2,
        summary="step 2 alt",
        next_note="Backtracked: applied your rank 2 alternative for step 2.",
    )

    contents = [m["content"] for m in thread.messages()]
    assert "answer step 2" in contents
    assert "answer step 3" not in contents
    assert "accepted:3" not in thread.checkpoints
    assert 4 not in thread.avoid
    assert "rank 2 alternative" in thread.preamble(3)
    assert thread.accepted_summaries[2] == "step 2 alt"


def test_compaction_keeps_root_summary_and_last_two_steps() -> None:
    thread = _thread_with_root()
    for step in range(1, 6):
        _accept(thread, step)

    before = thread.char_size()
    thread.compact(keep_last_steps=2)

    contents = [m["content"] for m in thread.messages()]
    assert contents[0] == "pre-analysis prompt"
    assert "step 1: A>>B1" in contents[2] and "step 3: A>>B3" in contents[2]
    assert "answer step 3" not in contents
    assert "answer step 4" in contents and "answer step 5" in contents
    assert thread.compactions == 1
    assert thread.needs_full_prompt() is True
    assert thread.char_size() <= before + 400
    assert thread.fork_to_step(5) == 0


def test_fork_into_compacted_range_rebuilds_summary_up_to_that_step() -> None:
    thread = _thread_with_root()
    for step in range(1, 6):
        _accept(thread, step)
    thread.compact(keep_last_steps=2)

    thread.fork_to_step(2)

    contents = [m["content"] for m in thread.messages()]
    assert len(contents) == 4
    assert "step 2: A>>B2" in contents[2]
    assert "step 3: A>>B3" not in contents[2]
    assert thread.needs_full_prompt() is True
    assert 3 not in thread.accepted_summaries


def test_should_compact_uses_budget() -> None:
    thread = ConversationThread(max_prompt_chars=50)
    thread.set_root([{"role": "user", "content": "x" * 60}])
    assert thread.should_compact() is True


def test_round_trip_dict() -> None:
    thread = _thread_with_root()
    _accept(thread, 1)
    thread.add_avoid(2, "bad")
    restored = ConversationThread.from_dict(thread.to_dict())
    assert restored.messages() == thread.messages()
    assert restored.checkpoints == thread.checkpoints
    assert restored.preamble(2) == thread.preamble(2)
    assert restored.accepted_summaries == thread.accepted_summaries


def test_empty_root_thread_still_works() -> None:
    thread = ConversationThread()
    thread.set_root([])
    assert thread.checkpoints["accepted:0"] == 0
    assert thread.messages() == []
    assert thread.needs_full_prompt() is True
