# Prompt Forest OpenClaw Plugin

This directory is a standalone OpenClaw plugin that wraps the Python `prompt_forest` package through a JSON bridge.

## Install locally into OpenClaw

```bash
openclaw plugins install ./extensions/prompt-forest -l
```

Then enable/configure `openclaw-prompt-forest` in `~/.openclaw/openclaw.json`:

```json5
{
  plugins: {
    allow: ["openclaw-prompt-forest"],
    entries: {
      "openclaw-prompt-forest": {
        enabled: true,
        config: {
          pythonBin: "/absolute/path/to/.venv/bin/python",
          projectRoot: "/absolute/path/to/openclaw_closedsourcemodel_RL",
          configPath: "configs/default.json",
          model: "gpt-4.1-mini",
          apiKeyEnv: "OPENAI_API_KEY",
          latencyMode: "full",
          visibility: "minimal",
          includeTrace: false
        }
      }
    }
  }
}
```

## Registered tools

- `prompt_forest_assist`
- `prompt_forest_feedback`
- `prompt_forest_state`

The bundled skill tells OpenClaw when to use the main tool automatically.
