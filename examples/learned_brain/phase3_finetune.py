#!/usr/bin/env python3
"""Phase 3 — Fine-tune a model on high-scoring emotional roleplay data.

Stage 1: Collect high-quality training data using gpt-5.4-mini + gpt-4o judge
Stage 2: Format data as JSONL for OpenAI fine-tuning API
Stage 3: Submit fine-tuning job
Stage 4: Evaluate fine-tuned model

Usage:
    cd examples
    OPENAI_API_KEY=... python -m learned_brain.phase3_finetune --stage 1
    OPENAI_API_KEY=... python -m learned_brain.phase3_finetune --stage 2
    OPENAI_API_KEY=... python -m learned_brain.phase3_finetune --stage 3
    OPENAI_API_KEY=... python -m learned_brain.phase3_finetune --stage 4
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openai
from brain_adaptive_prototype import PersonalityProfile
from brain_vs_plain_llm import BlindJudge, PlainLLMBaseline, _token_limit_kwargs
from brain_rl_evaluation import SCENARIOS
from learned_brain.learned_brain_engine import LearnedBrainEngine

DATA_DIR = Path(__file__).parent / "finetune_data"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# Multiple personality profiles for diverse training data
PERSONALITIES = [
    PersonalityProfile(
        name="Alex",
        background="32 years old, 8 years experience, underpaid for 2 years. Tired of being undervalued.",
        temperament="Hot-tempered, direct, takes disrespect personally. Speaks from the gut. Quick to escalate.",
        emotional_tendencies={
            "anger": "quick to flare, expressed openly",
            "patience": "runs out fast",
            "impulse": "high, speaks before thinking",
        },
    ),
    PersonalityProfile(
        name="Jordan",
        background="28 years old, first big career opportunity. Grew up having to fight for everything.",
        temperament="Anxious but determined, oscillates between confidence and self-doubt. Overthinks.",
        emotional_tendencies={
            "anxiety": "constant undercurrent, spikes under pressure",
            "determination": "fierce when pushed",
            "vulnerability": "hidden but present",
        },
    ),
    PersonalityProfile(
        name="Sam",
        background="45 years old, veteran professional, seen it all. Recently divorced, financially strained.",
        temperament="Calm exterior hiding deep frustration. Dry humor. Doesn't suffer fools. Quietly intense.",
        emotional_tendencies={
            "frustration": "slow burn, expressed through sarcasm",
            "resilience": "high, but wearing thin",
            "cynicism": "earned through experience",
        },
    ),
]


def _token_kw(model: str, limit: int) -> dict:
    """Return correct token limit kwarg for model."""
    if model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4"):
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}


def stage1_collect(
    gen_model: str = "gpt-5.4-mini",
    judge_model: str = "gpt-4o",
    n_rollouts_per_personality: int = 3,
    score_threshold: int = 25,
):
    """Stage 1: Collect high-quality training data.

    For each personality × scenario × rollout:
    - Generate responses with the best available model
    - Judge each with gpt-4o
    - Keep responses scoring >= threshold as fine-tuning targets
    """
    print("=" * 70)
    print("  STAGE 1: Collecting fine-tuning data")
    print(f"  Generator: {gen_model}")
    print(f"  Judge: {judge_model}")
    print(f"  Score threshold: {score_threshold}/30")
    print(f"  Personalities: {len(PERSONALITIES)}")
    print(f"  Rollouts per personality: {n_rollouts_per_personality}")
    print("=" * 70, flush=True)

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    brain = LearnedBrainEngine()
    judge = BlindJudge(model=judge_model)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_examples = []  # All scored examples
    high_quality = []  # Only >= threshold

    for pi, personality in enumerate(PERSONALITIES):
        plain = PlainLLMBaseline(personality, "")

        for si, scenario_data in enumerate(SCENARIOS):
            scenario_text = scenario_data["scenario"]
            turns = scenario_data["turns"]

            for ri in range(n_rollouts_per_personality):
                brain.reset(scenario_text)
                plain.reset_for_scenario(scenario_text)
                conversation_history: list[dict] = []
                conversation_lines = []

                for ti, turn in enumerate(turns):
                    event = turn["event"]
                    says = turn["says"]
                    event_desc = event.get("description", event.get("type", ""))

                    # Brain processes utterance
                    brain.process_utterance(says)
                    narrative = brain.get_emotional_narrative()

                    # Build system prompt with emotional context
                    system_prompt = f"""You are {personality.name}, a real human being in the following situation:

{scenario_text}

About you:
{personality.background}
Temperament: {personality.temperament}

YOUR EMOTIONAL STATE (how you're actually feeling right now):
{narrative}

RULES:
- You are a REAL PERSON, not an AI. Never break character.
- Respond naturally. Keep responses 1-3 sentences.
- Your emotional reactions should be realistic and consistent.
- Let your emotions BUILD across the conversation — don't reset each turn.
- If something makes you angry, STAY angry until something changes.
- React to what's said AND to the accumulated history of the conversation.
- Your personality is: {personality.temperament}
- Never say "I understand your perspective" or other AI-speak."""

                    messages = [{"role": "system", "content": system_prompt}]
                    messages.extend(conversation_history)
                    messages.append({"role": "user", "content": says})

                    # Generate with best model
                    try:
                        response = client.chat.completions.create(
                            model=gen_model,
                            messages=messages,
                            temperature=0.9,
                            **_token_kw(gen_model, 150),
                        )
                        speech = response.choices[0].message.content.strip().strip('"')
                        if speech.lower().startswith(personality.name.lower() + ":"):
                            speech = speech[len(personality.name) + 1:].strip().strip('"')
                    except Exception as e:
                        print(f"    Gen error: {e}", flush=True)
                        speech = "Look, this isn't working for me."

                    # Also generate plain baseline for comparison judge
                    plain_speech = plain.respond(says, event_desc)

                    # Judge
                    conv_summary = "\n".join(conversation_lines[-8:]) if conversation_lines else "(start)"
                    judgment = judge.judge(
                        scenario_text, personality, event_desc, says,
                        speech, plain_speech, conv_summary, ti + 1,
                    )

                    brain_scores = judgment["brain_scores"]
                    total = brain_scores.get("total", sum(
                        brain_scores.get(k, 0) for k in ["emotional_accuracy", "naturalness", "consistency"]
                    ))

                    example = {
                        "system_prompt": system_prompt,
                        "conversation_history": list(conversation_history),
                        "user_message": says,
                        "response": speech,
                        "score": total,
                        "scores_detail": brain_scores,
                        "personality": personality.name,
                        "scenario": scenario_data["name"],
                        "turn": ti + 1,
                        "emotional_narrative": narrative,
                    }
                    all_examples.append(example)

                    if total >= score_threshold:
                        high_quality.append(example)

                    print(f"    P:{personality.name} S:{si+1} R:{ri+1} T:{ti+1} "
                          f"score={total}/30 {'✓' if total >= score_threshold else '·'} "
                          f"({len(high_quality)} HQ / {len(all_examples)} total)", flush=True)

                    # Update conversation state
                    brain.process_utterance(speech)
                    conversation_history.append({"role": "user", "content": says})
                    conversation_history.append({"role": "assistant", "content": speech})
                    conversation_lines.append(f"  Them: \"{says[:60]}...\"")
                    conversation_lines.append(f"  {personality.name}: \"{speech[:60]}...\"")

    # Save
    with open(DATA_DIR / "all_scored_examples.pkl", "wb") as f:
        pickle.dump(all_examples, f)
    with open(DATA_DIR / "high_quality_examples.pkl", "wb") as f:
        pickle.dump(high_quality, f)

    scores = [e["score"] for e in all_examples]
    print(f"\n{'=' * 70}")
    print(f"  Collection complete!")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  High quality (>={score_threshold}): {len(high_quality)} ({len(high_quality)/len(all_examples)*100:.0f}%)")
    print(f"  Score distribution: mean={mean(scores):.1f}, min={min(scores)}, max={max(scores)}")
    print(f"  Saved to {DATA_DIR}")
    print(f"{'=' * 70}", flush=True)


def stage2_format(
    base_model: str = "gpt-4o-mini-2024-07-18",
    min_score: int = 24,
):
    """Stage 2: Format high-quality data as JSONL for OpenAI fine-tuning."""
    print("=" * 70)
    print("  STAGE 2: Formatting fine-tuning data")
    print(f"  Base model: {base_model}")
    print(f"  Min score: {min_score}")
    print("=" * 70, flush=True)

    with open(DATA_DIR / "all_scored_examples.pkl", "rb") as f:
        all_examples = pickle.load(f)

    # Filter by score
    good_examples = [e for e in all_examples if e["score"] >= min_score]
    print(f"  Examples >= {min_score}: {len(good_examples)} / {len(all_examples)}")

    if len(good_examples) < 10:
        print(f"  WARNING: Only {len(good_examples)} examples. Need at least 10 for fine-tuning.")
        print(f"  Try lowering min_score or collecting more data.")
        # Use everything above median
        median_score = sorted([e["score"] for e in all_examples])[len(all_examples) // 2]
        good_examples = [e for e in all_examples if e["score"] >= median_score]
        print(f"  Falling back to median cutoff ({median_score}): {len(good_examples)} examples")

    # Format as OpenAI fine-tuning JSONL
    jsonl_lines = []
    for ex in good_examples:
        messages = [{"role": "system", "content": ex["system_prompt"]}]
        messages.extend(ex["conversation_history"])
        messages.append({"role": "user", "content": ex["user_message"]})
        messages.append({"role": "assistant", "content": ex["response"]})

        jsonl_lines.append(json.dumps({"messages": messages}))

    # Split train/val (90/10)
    import random
    random.shuffle(jsonl_lines)
    n_val = max(1, len(jsonl_lines) // 10)
    val_lines = jsonl_lines[:n_val]
    train_lines = jsonl_lines[n_val:]

    train_path = DATA_DIR / "finetune_train.jsonl"
    val_path = DATA_DIR / "finetune_val.jsonl"

    with open(train_path, "w") as f:
        f.write("\n".join(train_lines))
    with open(val_path, "w") as f:
        f.write("\n".join(val_lines))

    print(f"  Train: {len(train_lines)} examples → {train_path}")
    print(f"  Val: {len(val_lines)} examples → {val_path}")

    # Token count estimate
    total_chars = sum(len(line) for line in train_lines)
    est_tokens = total_chars // 4
    print(f"  Estimated tokens: ~{est_tokens:,}")
    print(f"  Estimated cost: ~${est_tokens * 3 / 1_000_000:.2f} (gpt-4o-mini fine-tuning)")
    print(f"{'=' * 70}", flush=True)


def stage3_submit(
    base_model: str = "gpt-4o-mini-2024-07-18",
    n_epochs: int = 3,
    suffix: str = "emotional-roleplay",
):
    """Stage 3: Submit fine-tuning job to OpenAI."""
    print("=" * 70)
    print("  STAGE 3: Submitting fine-tuning job")
    print(f"  Base model: {base_model}")
    print(f"  Epochs: {n_epochs}")
    print("=" * 70, flush=True)

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    train_path = DATA_DIR / "finetune_train.jsonl"
    val_path = DATA_DIR / "finetune_val.jsonl"

    # Upload training file
    print("  Uploading training file...", flush=True)
    with open(train_path, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    print(f"  Train file ID: {train_file.id}")

    # Upload validation file
    print("  Uploading validation file...", flush=True)
    with open(val_path, "rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")
    print(f"  Val file ID: {val_file.id}")

    # Create fine-tuning job
    print("  Creating fine-tuning job...", flush=True)
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model=base_model,
        suffix=suffix,
        hyperparameters={
            "n_epochs": n_epochs,
        },
    )

    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")

    # Save job info
    job_info = {
        "job_id": job.id,
        "base_model": base_model,
        "status": job.status,
        "train_file": train_file.id,
        "val_file": val_file.id,
        "created_at": time.time(),
    }
    with open(DATA_DIR / "finetune_job.json", "w") as f:
        json.dump(job_info, f, indent=2)

    print(f"\n  Job submitted! Monitor with:")
    print(f"    python -m learned_brain.phase3_finetune --status")
    print(f"{'=' * 70}", flush=True)


def check_status():
    """Check fine-tuning job status."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    job_path = DATA_DIR / "finetune_job.json"
    if not job_path.exists():
        print("No fine-tuning job found. Run --stage 3 first.")
        return

    with open(job_path) as f:
        job_info = json.load(f)

    job = client.fine_tuning.jobs.retrieve(job_info["job_id"])
    print(f"Job: {job.id}")
    print(f"Status: {job.status}")
    print(f"Base model: {job.model}")

    if job.fine_tuned_model:
        print(f"Fine-tuned model: {job.fine_tuned_model}")
        # Save the model name
        job_info["fine_tuned_model"] = job.fine_tuned_model
        with open(job_path, "w") as f:
            json.dump(job_info, f, indent=2)

    if job.status == "failed":
        print(f"Error: {job.error}")

    # Show recent events
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=10)
    print("\nRecent events:")
    for event in reversed(events.data):
        print(f"  [{event.created_at}] {event.message}")


def stage4_eval(n_trials: int = 3):
    """Stage 4: Evaluate the fine-tuned model vs plain LLM."""
    print("=" * 70)
    print("  STAGE 4: Evaluating fine-tuned model")
    print("=" * 70, flush=True)

    job_path = DATA_DIR / "finetune_job.json"
    if not job_path.exists():
        print("No fine-tuning job found. Run --stage 3 first.")
        return

    with open(job_path) as f:
        job_info = json.load(f)

    ft_model = job_info.get("fine_tuned_model")
    if not ft_model:
        # Try to get it from the API
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        job = client.fine_tuning.jobs.retrieve(job_info["job_id"])
        ft_model = job.fine_tuned_model
        if not ft_model:
            print(f"Fine-tuning not complete yet. Status: {job.status}")
            return
        job_info["fine_tuned_model"] = ft_model
        with open(job_path, "w") as f:
            json.dump(job_info, f, indent=2)

    print(f"  Fine-tuned model: {ft_model}")
    print(f"  Trials: {n_trials}")

    # Set the gen model to our fine-tuned model and run eval
    os.environ["GEN_MODEL"] = ft_model

    from learned_brain.eval_phase2 import run_phase2_eval

    personality = PERSONALITIES[0]  # Alex
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o")

    all_p2, all_p = [], []
    for t in range(n_trials):
        print(f"\n{'#' * 70}")
        print(f"  TRIAL {t+1}/{n_trials}")
        print(f"{'#' * 70}", flush=True)
        p2, p = run_phase2_eval(personality, SCENARIOS)
        all_p2.append(p2)
        all_p.append(p)

    if n_trials > 1:
        from statistics import stdev
        print(f"\n{'█' * 70}")
        print(f"  AGGREGATE ({n_trials} trials)")
        print(f"{'█' * 70}")
        print(f"  Fine-tuned: {mean(all_p2):.1f}/30  (std: {stdev(all_p2):.1f})")
        print(f"  Plain:      {mean(all_p):.1f}/30  (std: {stdev(all_p):.1f})")
        improvement = ((mean(all_p2) - mean(all_p)) / mean(all_p)) * 100
        print(f"  Improvement: {improvement:+.1f}%  {'✓ TARGET HIT!' if improvement >= 20 else '✗ not yet'}")
        print(f"{'█' * 70}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Fine-tuning Pipeline")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run specific stage (1/2/3/4) or 0 for 1+2+3")
    parser.add_argument("--gen-model", type=str, default="gpt-5.4-mini",
                        help="Model for generating training data")
    parser.add_argument("--judge-model", type=str, default="gpt-4o",
                        help="Judge model for scoring")
    parser.add_argument("--base-model", type=str, default="gpt-4o-mini-2024-07-18",
                        help="Base model to fine-tune")
    parser.add_argument("--rollouts", type=int, default=3,
                        help="Rollouts per personality")
    parser.add_argument("--min-score", type=int, default=24,
                        help="Minimum score for fine-tuning data")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Fine-tuning epochs")
    parser.add_argument("--trials", type=int, default=3,
                        help="Evaluation trials")
    parser.add_argument("--status", action="store_true",
                        help="Check fine-tuning job status")
    args = parser.parse_args()

    if args.status:
        check_status()
        return

    print(f"\n{'#' * 70}")
    print(f"  PHASE 3: FINE-TUNING PIPELINE")
    print(f"  Stage: {'all (1-3)' if args.stage == 0 else args.stage}")
    print(f"{'#' * 70}", flush=True)

    if args.stage == 0 or args.stage == 1:
        stage1_collect(
            gen_model=args.gen_model,
            judge_model=args.judge_model,
            n_rollouts_per_personality=args.rollouts,
            score_threshold=args.min_score,
        )

    if args.stage == 0 or args.stage == 2:
        stage2_format(base_model=args.base_model, min_score=args.min_score)

    if args.stage == 0 or args.stage == 3:
        stage3_submit(base_model=args.base_model, n_epochs=args.epochs)

    if args.stage == 4:
        stage4_eval(n_trials=args.trials)

    if args.stage != 4:
        print(f"\n{'#' * 70}")
        print("  STAGES 1-3 COMPLETE")
        print("  Monitor fine-tuning: python -m learned_brain.phase3_finetune --status")
        print("  Evaluate: python -m learned_brain.phase3_finetune --stage 4")
        print(f"{'#' * 70}", flush=True)


if __name__ == "__main__":
    main()
