"""One chat thread for the mechanism loop (``conversation_mode: threaded``).

The thread starts from a root (the unified pre-analysis exchange, or nothing)
and grows by one user/assistant exchange per proposal. ``accepted:<k>`` marks
the turn count right after step k was accepted, so a failed proposal for step
k+1 is dropped by forking back to ``accepted:<k>``; what was wrong with it
survives only as a short avoid note on the next turn for that step.

Assistant turns are text (``serialise_chat_messages`` keeps only role and
content), so the thread is a plain list of ``{"role", "content"}`` messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

THREAD_SCHEMA = "conversation_thread.v1"
MAX_AVOID_NOTES = 6
MAX_NOTE_CHARS = 300
DEFAULT_MAX_PROMPT_CHARS = 120_000


def _clip(text: str, limit: int = MAX_NOTE_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _int_keyed(raw: Any) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                out[int(key)] = value
            except (TypeError, ValueError):
                continue
    return out


@dataclass
class ConversationThread:
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS
    keep_last_steps: int = 2
    turns: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: Dict[str, int] = field(default_factory=dict)
    avoid: Dict[int, List[str]] = field(default_factory=dict)
    step_notes: Dict[int, List[str]] = field(default_factory=dict)
    accepted_summaries: Dict[int, str] = field(default_factory=dict)
    root_len: int = 0
    compacted_through: int = 0
    compactions: int = 0

    # ------------------------------------------------------------------ build
    def set_root(self, messages: List[Dict[str, Any]]) -> None:
        self.turns = [
            {"role": str(m.get("role")), "content": str(m.get("content") or ""), "kind": "root", "step": 0}
            for m in messages
            if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
        ]
        self.root_len = len(self.turns)
        self.checkpoints = {"accepted:0": self.root_len}
        self.compacted_through = 0

    def append_exchange(
        self,
        step: int,
        user: str,
        assistant: str,
        *,
        full: bool,
        guidance_sha: Optional[str] = None,
    ) -> None:
        self.turns.append(
            {
                "role": "user",
                "content": user,
                "kind": "proposal",
                "step": int(step),
                "full": bool(full),
                "guidance_sha": guidance_sha or "",
            }
        )
        self.turns.append({"role": "assistant", "content": assistant, "kind": "proposal", "step": int(step)})

    def last_guidance_sha(self) -> str:
        """Hash of the harness guidance in the latest retained turn that sent it."""
        for turn in reversed(self.turns):
            if turn.get("role") == "user" and turn.get("guidance_sha"):
                return str(turn["guidance_sha"])
        return ""

    def on_accept(self, step: int, *, rank: Optional[int], summary: str, next_note: str) -> None:
        step = int(step)
        self.checkpoints[f"accepted:{step}"] = len(self.turns)
        self._drop_after(step)
        self.accepted_summaries[step] = str(summary)
        self.avoid.pop(step, None)
        self.step_notes.pop(step, None)
        if next_note:
            self.step_notes[step + 1] = [_clip(next_note, 900)]

    def add_avoid(self, step: int, note: str) -> None:
        note = _clip(note)
        if not note:
            return
        notes = self.avoid.setdefault(int(step), [])
        if note in notes:
            return
        notes.append(note)
        del notes[:-MAX_AVOID_NOTES]

    def add_note(self, step: int, note: str) -> None:
        note = _clip(note, 900)
        if note:
            self.step_notes.setdefault(int(step), []).append(note)

    # ----------------------------------------------------------------- query
    def messages(self) -> List[Dict[str, str]]:
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def char_size(self) -> int:
        return sum(len(t.get("content") or "") for t in self.turns)

    def should_compact(self) -> bool:
        return self.char_size() > self.max_prompt_chars

    def needs_full_prompt(self) -> bool:
        return not any(t.get("full") for t in self.turns if t.get("role") == "user")

    def preamble(self, step: int) -> str:
        lines: List[str] = list(self.step_notes.get(int(step)) or [])
        avoid = self.avoid.get(int(step)) or []
        if avoid:
            lines.append("Already rejected for this step (do not propose these again):")
            lines.extend(f"- {note}" for note in avoid)
        return "\n".join(lines)

    def accepted_steps(self) -> List[int]:
        steps = []
        for key in self.checkpoints:
            if key.startswith("accepted:"):
                value = int(key.split(":", 1)[1])
                if value > 0:
                    steps.append(value)
        return sorted(steps)

    # ---------------------------------------------------------------- rewind
    def fork_to_step(self, step: int) -> int:
        """Truncate the thread to just after step ``step`` was accepted.

        Returns the number of turns dropped.
        """
        step = max(0, int(step))
        before = len(self.turns)
        if 0 < step < self.compacted_through:
            self._rebuild_summary_root(step)
        else:
            target = self.checkpoints.get(f"accepted:{step}")
            if target is None:
                earlier = [s for s in [0, *self.accepted_steps()] if s < step]
                target = self.checkpoints.get(f"accepted:{max(earlier) if earlier else 0}", self.root_len)
            del self.turns[target:]
        self._drop_after(step)
        for later in [s for s in list(self.avoid) if s > step + 1]:
            self.avoid.pop(later, None)
        for later in [s for s in list(self.step_notes) if s > step + 1]:
            self.step_notes.pop(later, None)
        return max(0, before - len(self.turns))

    def _drop_after(self, step: int) -> None:
        for key in list(self.checkpoints):
            if key.startswith("accepted:") and int(key.split(":", 1)[1]) > step:
                self.checkpoints.pop(key)
        for later in [s for s in self.accepted_summaries if s > step]:
            self.accepted_summaries.pop(later, None)

    # ------------------------------------------------------------ compaction
    def _summary_pair(self, through: int) -> List[Dict[str, Any]]:
        lines = [
            self.accepted_summaries.get(s, f"step {s}: (no summary)")
            for s in range(1, through + 1)
        ]
        user = (
            "Summary of the validated mechanism so far (earlier turns were compacted):\n"
            + "\n".join(lines)
            + "\nContinue from the state given in the next message."
        )
        return [
            {"role": "user", "content": user, "kind": "summary", "step": through},
            {
                "role": "assistant",
                "content": "Understood. I will continue the mechanism from the latest state.",
                "kind": "summary",
                "step": through,
            },
        ]

    def _rebuild_summary_root(self, through: int) -> None:
        self.turns = self.turns[: self.root_len] + self._summary_pair(through)
        self.compacted_through = through
        for s in range(1, through + 1):
            self.checkpoints[f"accepted:{s}"] = len(self.turns)

    def compact(self, keep_last_steps: Optional[int] = None) -> bool:
        keep = self.keep_last_steps if keep_last_steps is None else int(keep_last_steps)
        accepted = self.accepted_steps()
        if not accepted:
            return False
        cut = max(accepted) - max(0, keep)
        if cut <= self.compacted_through:
            return False
        retained = [t for t in self.turns[self.root_len :] if t.get("kind") == "proposal" and int(t.get("step") or 0) > cut]
        self.turns = self.turns[: self.root_len] + self._summary_pair(cut) + retained
        self.compacted_through = cut
        for s in accepted:
            if s <= cut:
                self.checkpoints[f"accepted:{s}"] = self.root_len + 2
            else:
                last = max(
                    (i for i, t in enumerate(self.turns) if t.get("kind") == "proposal" and int(t.get("step") or 0) == s),
                    default=None,
                )
                if last is not None:
                    self.checkpoints[f"accepted:{s}"] = last + 1
        self.compactions += 1
        return True

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": THREAD_SCHEMA,
            "max_prompt_chars": self.max_prompt_chars,
            "keep_last_steps": self.keep_last_steps,
            "turns": [dict(t) for t in self.turns],
            "checkpoints": dict(self.checkpoints),
            "avoid": {str(k): list(v) for k, v in self.avoid.items()},
            "step_notes": {str(k): list(v) for k, v in self.step_notes.items()},
            "accepted_summaries": {str(k): v for k, v in self.accepted_summaries.items()},
            "root_len": self.root_len,
            "compacted_through": self.compacted_through,
            "compactions": self.compactions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationThread":
        data = data if isinstance(data, dict) else {}
        return cls(
            max_prompt_chars=int(data.get("max_prompt_chars") or DEFAULT_MAX_PROMPT_CHARS),
            keep_last_steps=int(data.get("keep_last_steps") if data.get("keep_last_steps") is not None else 2),
            turns=[dict(t) for t in data.get("turns") or [] if isinstance(t, dict)],
            checkpoints={str(k): int(v) for k, v in (data.get("checkpoints") or {}).items()},
            avoid={k: list(v) for k, v in _int_keyed(data.get("avoid")).items()},
            step_notes={k: list(v) for k, v in _int_keyed(data.get("step_notes")).items()},
            accepted_summaries={k: str(v) for k, v in _int_keyed(data.get("accepted_summaries")).items()},
            root_len=int(data.get("root_len") or 0),
            compacted_through=int(data.get("compacted_through") or 0),
            compactions=int(data.get("compactions") or 0),
        )
