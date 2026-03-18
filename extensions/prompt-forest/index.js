import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PLUGIN_ID = "openclaw-prompt-forest";
const DEFAULT_VISIBILITY = "minimal";
const DEFAULT_LATENCY_MODE = "full";

function pluginRoot() {
  return path.dirname(fileURLToPath(import.meta.url));
}

function readPluginConfig(api) {
  const entries = api?.config?.plugins?.entries ?? {};
  const config = entries[PLUGIN_ID]?.config ?? {};
  return config && typeof config === "object" ? config : {};
}

function mergePaths(left, right) {
  if (left && right) {
    return `${left}${path.delimiter}${right}`;
  }
  return left || right || "";
}

function resolveProjectRoot(config) {
  const root = String(config.projectRoot ?? "").trim();
  if (root) {
    return path.resolve(root);
  }
  return path.resolve(pluginRoot(), "..", "..");
}

function resolveConfigPath(config, projectRootPath) {
  const configured = String(config.configPath ?? "").trim();
  if (!configured) {
    return path.join(projectRootPath, "configs", "default.json");
  }
  return path.isAbsolute(configured) ? configured : path.join(projectRootPath, configured);
}

function resolvePythonBin(config) {
  const pythonBin = String(config.pythonBin ?? "").trim();
  return pythonBin || "python3";
}

function formatNumber(value) {
  return typeof value === "number" ? value.toFixed(3) : "n/a";
}

function formatPaths(paths) {
  if (!Array.isArray(paths) || paths.length === 0) {
    return "";
  }
  return paths
    .filter((entry) => Array.isArray(entry) && entry.length > 0)
    .map((entry) => entry.join(" -> "))
    .join(" | ");
}

function formatRunText(result) {
  const lines = [];
  if (result.answer) {
    lines.push(result.answer);
    lines.push("");
  }
  lines.push("Prompt Forest meta:");
  lines.push(`task_type=${result.task_type || "auto"}`);
  lines.push(`selected_branch=${result.selected_branch || "n/a"}`);
  lines.push(`reward=${formatNumber(result.reward)} confidence=${formatNumber(result.confidence)}`);
  if (result.failure_reason) {
    lines.push(`failure_reason=${result.failure_reason}`);
  }
  const pathText = formatPaths(result.activated_paths);
  if (pathText) {
    lines.push(`activated_paths=${pathText}`);
  }
  if (result.trace) {
    lines.push("");
    lines.push("Prompt Forest trace:");
    lines.push(result.trace);
  }
  return lines.join("\n");
}

function formatFeedbackText(result) {
  if (result.ok) {
    return [
      "Prompt Forest feedback applied.",
      `task_id=${result.task_id || "n/a"}`,
      `old_reward=${formatNumber(result.old_reward)}`,
      `new_reward=${formatNumber(result.new_reward)}`,
    ].join("\n");
  }
  return JSON.stringify(result, null, 2);
}

function formatStateText(result) {
  const memory = result.memory ?? {};
  return [
    "Prompt Forest state snapshot.",
    `branch_count=${Object.keys(result.branches ?? {}).length}`,
    `memory_records=${memory.records ?? "n/a"}`,
    `user_profiles=${memory.user_profiles ?? "n/a"}`,
  ].join("\n");
}

function buildCommonPayload(config, params) {
  const payload = {
    artifacts_dir: params.artifactsDir ?? config.artifactsDir,
    model: params.model ?? config.model,
    api_key_env: params.apiKeyEnv ?? config.apiKeyEnv,
    base_url: params.baseUrl ?? config.baseUrl,
    api_mode: params.apiMode ?? config.apiMode,
    reasoning_effort: params.reasoningEffort ?? config.reasoningEffort,
    temperature: params.temperature ?? config.temperature,
    max_output_tokens: params.maxOutputTokens ?? config.maxOutputTokens,
    latency_mode: params.latencyMode ?? config.latencyMode ?? DEFAULT_LATENCY_MODE,
    visibility: params.visibility ?? config.visibility ?? DEFAULT_VISIBILITY,
    include_trace: params.includeTrace ?? config.includeTrace ?? false,
  };
  return Object.fromEntries(Object.entries(payload).filter(([, value]) => value !== undefined && value !== ""));
}

function runBridge(command, payload, config) {
  const projectRootPath = resolveProjectRoot(config);
  const configPath = resolveConfigPath(config, projectRootPath);
  const pythonBin = resolvePythonBin(config);
  const child = spawn(
    pythonBin,
    ["-m", "prompt_forest.openclaw_bridge", "--config", configPath, command],
    {
      cwd: projectRootPath,
      env: {
        ...process.env,
        PYTHONPATH: mergePaths(path.join(projectRootPath, "src"), process.env.PYTHONPATH ?? ""),
      },
      stdio: ["pipe", "pipe", "pipe"],
    }
  );

  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", (error) => {
      reject(error);
    });
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || stdout.trim() || `Prompt Forest bridge failed with code ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`Prompt Forest bridge returned invalid JSON: ${stdout}\n${error}`));
      }
    });

    child.stdin.end(JSON.stringify(payload));
  });
}

export default function registerPromptForest(api) {
  api.registerTool(
    {
      name: "prompt_forest_assist",
      description:
        "Run a user task through Prompt Forest for planning, verification, critique, code review, and structured multi-branch reasoning.",
      parameters: {
        type: "object",
        additionalProperties: false,
        required: ["task"],
        properties: {
          task: { type: "string", description: "User request to route through Prompt Forest." },
          taskType: {
            type: "string",
            enum: ["auto", "math", "planning", "factual", "code", "creative", "general"],
            description: "Optional explicit task type. Use auto when unsure."
          },
          userId: { type: "string", description: "Stable user id for memory personalization." },
          metadata: { type: "object", additionalProperties: true, description: "Additional Prompt Forest metadata." },
          contextSeed: { type: "string", description: "Concise context to prepend before branch execution." },
          expectedKeywords: {
            type: "array",
            items: { type: "string" },
            description: "Keywords the final answer should likely cover."
          },
          requiredChecks: {
            type: "array",
            items: { type: "string" },
            description: "Required substrings or checks that must appear in the final answer."
          },
          visibility: {
            type: "string",
            enum: ["minimal", "eval", "opt", "full"],
            description: "Trace verbosity in the returned metadata."
          },
          latencyMode: {
            type: "string",
            enum: ["full", "fast"],
            description: "Fast disables some adaptive passes for lower latency."
          },
          includeTrace: { type: "boolean", description: "Include a human-readable Prompt Forest trace." }
        }
      },
      async execute(_id, params) {
        const config = readPluginConfig(api);
        const payload = {
          ...buildCommonPayload(config, params),
          task: params.task,
          task_type: params.taskType ?? "auto",
          user_id: params.userId,
          metadata: params.metadata,
          context_seed: params.contextSeed,
          expected_keywords: params.expectedKeywords,
          required_checks: params.requiredChecks,
        };
        const result = await runBridge("run", payload, config);
        return {
          content: [
            {
              type: "text",
              text: formatRunText(result),
            },
          ],
        };
      },
    },
    { optional: false }
  );

  api.registerTool(
    {
      name: "prompt_forest_feedback",
      description:
        "Apply explicit user feedback or a corrected answer back into Prompt Forest so later routing improves.",
      parameters: {
        type: "object",
        additionalProperties: false,
        required: ["taskId", "score"],
        properties: {
          taskId: { type: "string", description: "Prompt Forest task id returned by prompt_forest_assist." },
          score: { type: "number", description: "Feedback score in [0,1] or [1,5]." },
          accepted: { type: "boolean", description: "Mark the answer as explicitly accepted." },
          rejected: { type: "boolean", description: "Mark the answer as explicitly rejected." },
          correctedAnswer: { type: "string", description: "Corrected answer from the user, if any." },
          feedbackText: { type: "string", description: "Natural-language feedback from the user." },
          userId: { type: "string", description: "Stable user id for personalized memory." }
        }
      },
      async execute(_id, params) {
        const config = readPluginConfig(api);
        const payload = {
          ...buildCommonPayload(config, params),
          task_id: params.taskId,
          score: params.score,
          accepted: params.accepted,
          rejected: params.rejected,
          corrected_answer: params.correctedAnswer,
          feedback_text: params.feedbackText,
          user_id: params.userId,
        };
        const result = await runBridge("feedback", payload, config);
        return {
          content: [
            {
              type: "text",
              text: formatFeedbackText(result),
            },
          ],
        };
      },
    },
    { optional: true }
  );

  api.registerTool(
    {
      name: "prompt_forest_state",
      description: "Inspect Prompt Forest branch and memory state for debugging or evaluation.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          includeTrace: { type: "boolean" }
        }
      },
      async execute(_id, params) {
        const config = readPluginConfig(api);
        const result = await runBridge("state", buildCommonPayload(config, params), config);
        return {
          content: [
            {
              type: "text",
              text: formatStateText(result),
            },
          ],
        };
      },
    },
    { optional: true }
  );
}
