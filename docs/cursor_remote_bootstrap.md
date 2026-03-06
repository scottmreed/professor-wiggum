# Cursor Remote Bootstrap Troubleshooting

## Symptom

During Cursor Remote startup, the agent repeatedly fails with:

`Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'extensionless' imported from /workspace/`

This happens while installing or starting Cursor's `exec-daemon`.

## Why It Happens

This is a Node bootstrap issue on the remote host, usually caused by a global `NODE_OPTIONS` preload (for example an `--import extensionless` hook) that runs before this repository's Python app starts.

## Safe Fix (Recommended)

Run in the remote shell:

```bash
echo "$NODE_OPTIONS"
unset NODE_OPTIONS
```

Then reopen/reconnect the Cursor remote workspace.

## If You Must Keep `NODE_OPTIONS`

Install the missing preload package in the workspace:

```bash
npm install --save-dev extensionless
```

This repo is Python-first; adding the package only supports Node preload compatibility for remote tooling.

## Verification

After reconnecting:

1. Cursor agent starts without repeated `exec-daemon` failures.
2. `python main.py serve` starts normally.
3. The UI loads at `http://127.0.0.1:8010/ui/`.
