# Agent Bridge — keyless runs with a subagent as the model

The agent bridge lets you run the harness (single reactions, eval tiers,
curriculum batches) **without any provider API key**, by having an external
agent or subagent answer each model call. It exists so contributors whose setup
is an agent loop — not a hosted API key — can still produce runs, traces, and
evidence.

It is a normal entry in the model catalog (`mechanistic_agent/model_pricing.json`):

```
agent-bridge   provider=agent_bridge   supports_tools=true   pricing=0
```

and is routed by `mechanistic_agent.llm.get_chat_model`, exactly like the OpenAI,
Anthropic, and Gemini providers.

## Privacy contract (the important part)

The whole runtime reaches a model through one seam:
`get_chat_model(model).invoke(messages, tools=..., tool_choice=...)`. The bridge
serialises **only** those three things into each request, under a `model_input`
block:

```json
{
  "schema": "mechanistic.agent_bridge/request@1",
  "request_id": "…",
  "model": "agent-bridge",
  "model_input": { "messages": [...], "tools": [...], "tool_choice": {...} }
}
```

`messages` are serialised with the *same* `serialise_chat_messages` helper the
OpenAI adapter uses, so the responder sees byte-for-byte the message view a
keyed model would — **and nothing else**. No run state, eval ground truth,
atom-map context, or scoring information is added (the harness already strips
privileged context before this seam; see
`tests/fast/test_tool_executor_does_not_forward_raw_mapped_prompt_context`). This
invariant is enforced by `tests/fast/test_agent_bridge.py`.

## Enabling it

```bash
export MECHANISTIC_AGENT_BRIDGE_DIR=.agent_bridge      # exchange directory
# optional:
export MECHANISTIC_AGENT_BRIDGE_TIMEOUT=1800           # seconds per call (fail loud)
export MECHANISTIC_AGENT_BRIDGE_POLL_SECONDS=0.2       # poll interval

# run any harness command with the bridge model:
python main.py run --starting "CCBr.[Cl-]" --products "CCCl.[Br-]" --model agent-bridge
# or force it for every step:
MECHANISTIC_ACTIVE_MODEL=agent-bridge python main.py eval --tier easy --model agent-bridge
```

No API key is required. `get_model_api_key("agent-bridge")` returns a non-empty
sentinel so the tools that gate on "is a key configured?" treat the bridge as
available; the value is never used as a credential.

## Protocol

Exchange directory layout:

```
<MECHANISTIC_AGENT_BRIDGE_DIR>/
  requests/<seq>-<uuid>.json     # written by the harness (the model_input)
  responses/<seq>-<uuid>.json    # written by the responder (the tool call)
```

A response file is matched to its request by identical basename and looks like:

```json
{
  "tool_calls": [
    { "name": "<forced function name>", "arguments": { ... } }
  ],
  "content": "",
  "usage": { "input_tokens": 0, "output_tokens": 0 }
}
```

`arguments` may be a JSON object (it is encoded to a string automatically) or a
JSON string. `usage` is optional.

## Writing a responder (subagent loop)

`mechanistic_agent.agent_bridge` ships small helpers so a responder never has to
poke files by hand and never reads anything beyond the request:

```python
import time
from mechanistic_agent.agent_bridge import pending_requests, read_request, write_response

BRIDGE_DIR = ".agent_bridge"

while True:
    for req_path in pending_requests(BRIDGE_DIR):
        model_input = read_request(req_path)["model_input"]
        # Hand model_input["messages"] / ["tools"] / ["tool_choice"] to your
        # agent/subagent. Produce the arguments object for the forced tool.
        args = my_agent_answer(model_input)          # <- your model stands in here
        write_response(req_path, tool_calls=[{
            "name": model_input["tool_choice"]["function"]["name"],
            "arguments": args,
        }])
    time.sleep(0.2)
```

## Reproducibility / CI replay

Because requests and responses are plain files keyed by basename, you can
**pre-seed** `responses/` to replay a run deterministically with no agent and no
keys — useful for CI. If a response never arrives within the timeout the adapter
raises (no silent degradation), consistent with SOUL.md Guardrail 5.

## Attribution

Runs made through the bridge are attributed to the `agent-bridge` model
(`model_family: "agent"`, pricing 0) in traces and the leaderboard, so they are
never misattributed to a hosted model.
