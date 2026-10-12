"""build a labeled writing-style dataset through nvidia's openai-compatible api"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

# how this file works
# 1 read the questions and the three model names
# 2 load the api key from .env without showing it
# 3 ask each model the same question
# 4 save each finished answer so the run can resume later


BASE_URL = "https://integrate.api.nvidia.com/v1"
# keep the shared writing instruction here so every model gets the same request
SYSTEM_INSTRUCTION = "Answer directly in 80 to 120 words. Do not include analysis or reasoning."
# use these settings to keep each model's internal reasoning short enough to leave room for its answer
MODEL_REQUEST_OPTIONS = {
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "nvidia/nemotron-3.5-lightning-30b-a3b": {"extra_body": {"reasoning_budget": 128}},
    "z-ai/glm-5.3-flash": {"reasoning_effort": "low", "extra_body": {"chat_template_kwargs": {"clear_thinking": True}}},
}
# name the columns that must exist in the input files before i make api calls
PROMPT_FIELDS = {"prompt_id", "category", "split", "LLM_Input"}
MODEL_FIELDS = {"model_id", "family", "provider"}
# list every column i will write to the finished dataset including the assignment's three required fields
OUTPUT_FIELDS = [
    "LLM_name", "LLM_Input", "LLM_output", "prompt_id", "category", "split",
    "model_id", "family", "provider", "prompt_sha256", "collected_at_utc",
    "temperature", "max_tokens", "response_id", "finish_reason",
    "prompt_tokens", "completion_tokens", "system_instruction",
]


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    """read a csv and make sure its header contains the required columns"""
    # some csv programs add an extra marker at the start so accept that too
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # skip comment lines such as the guidance in models.example.csv
        rows = [line for line in handle if not line.lstrip().startswith("#")]
    # stop early if there is no header or data to parse
    if not rows:
        raise ValueError(f"{path} is empty")
    # convert each row into a dictionary so the rest of the code can use column names
    reader = csv.DictReader(rows)
    missing = required - set(reader.fieldnames or [])
    # name missing columns in the error so the file is easy to fix
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return list(reader)


def load_prompts(path: Path) -> list[dict[str, str]]:
    """load questions and reject blank or repeated prompt ids"""
    prompts = read_csv(path, PROMPT_FIELDS)
    # track ids because each id must identify exactly one question
    ids: set[str] = set()
    for row in prompts:
        # strip spaces so an accidental space does not create a different id
        prompt_id = (row["prompt_id"] or "").strip()
        if not prompt_id or not (row["LLM_Input"] or "").strip():
            raise ValueError("Every prompt needs a prompt_id and LLM_Input")
        # reject duplicates because the resume logic keys saved answers by prompt id
        if prompt_id in ids:
            raise ValueError(f"Duplicate prompt_id: {prompt_id}")
        ids.add(prompt_id)
    return prompts


def load_models(path: Path) -> list[dict[str, str]]:
    """load the chosen models and require distinct ids and family labels"""
    models = read_csv(path, MODEL_FIELDS)
    if not models:
        raise ValueError(f"No models selected in {path}")
    ids: set[str] = set()
    families: set[str] = set()
    for row in models:
        # require all three pieces of model metadata for every selection
        for field in MODEL_FIELDS:
            row[field] = (row[field] or "").strip()
            if not row[field]:
                raise ValueError(f"Every model needs {field}")
        # prevent two versions of one labeled family from counting as two families
        if row["model_id"] in ids or row["family"].casefold() in families:
            raise ValueError("Model IDs and family labels must each be unique")
        ids.add(row["model_id"])
        families.add(row["family"].casefold())
    return models


def client_from_env() -> OpenAI:
    """read the private key locally and build a client pointed at nvidia"""
    # load .env without printing saving or passing its key on the command line
    load_dotenv()
    key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
    if not key or key == "replace_with_your_key":
        raise ValueError("Set NVIDIA_API_KEY in .env or your environment first")
    # retry here so the terminal shows when another request is being made
    return OpenAI(base_url=BASE_URL, api_key=key, max_retries=0, timeout=60.0)


def available_ids(client: OpenAI) -> set[str]:
    """ask nvidia which model ids this key can see right now"""
    return {model.id for model in client.models.list().data}


def verify_ids(models: list[dict[str, str]], available: set[str]) -> None:
    """reject selected ids that are absent from the account's model list"""
    missing = [row["model_id"] for row in models if row["model_id"] not in available]
    if missing:
        raise ValueError("Model IDs absent from this account's list: " + ", ".join(missing))


def chat_with_retries(client: OpenAI, model_id: str, prompt: str,
                      temperature: float, max_tokens: int, attempts: int = 5):
    """retry temporary api failures, but let permanent errors stop the run"""
    # make one request at a time using the same writing instruction for each family
    for attempt in range(attempts):
        try:
            # add per-model settings only when that model needs them
            return client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                **MODEL_REQUEST_OPTIONS.get(model_id, {}),
            )
        except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as exc:
            # do not retry most client errors because changing time will not fix a bad request
            if isinstance(exc, APIStatusError) and exc.status_code < 500 and not isinstance(exc, RateLimitError):
                raise
            # give up after the configured number of tries instead of looping forever
            if attempt == attempts - 1:
                raise
            # wait a little longer each time to respect rate limits and server recovery
            delay = min(30.0, 2 ** attempt + random.uniform(0, 0.5))
            print(f"Temporary API error; retrying in {delay:.1f}s ({attempt + 1}/{attempts - 1})", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("Unreachable retry state")


def response_text(response) -> str:
    """accept only a complete, nonempty final answer as a dataset sample"""
    if not response.choices:
        raise ValueError("API response has no choices")
    # reject a response that ended only because the token limit was reached
    if response.choices[0].finish_reason == "length":
        raise ValueError("API response was truncated; raise --max-tokens and retry")
    # keep the public answer text and leave any separate reasoning trace out of the dataset
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("API response has no text output")
    return content


def load_completed(path: Path, prompts: list[dict[str, str]]) -> set[tuple[str, str]]:
    """find prompt/model pairs already saved so a rerun can resume safely"""
    if not path.exists():
        return set()
    # make a hash of each question so edits made after collection are noticed
    known_hashes = {
        row["prompt_id"]: hashlib.sha256(row["LLM_Input"].encode("utf-8")).hexdigest()
        for row in prompts
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    # refuse to append rows if the existing csv uses another column layout
    if header != OUTPUT_FIELDS:
        raise ValueError(f"{path} has a different column layout; use a new output file")
    rows = read_csv(path, {"prompt_id", "model_id", "prompt_sha256"})
    completed: set[tuple[str, str]] = set()
    for row in rows:
        prompt_id = row["prompt_id"]
        # avoid silently mixing answers to different wording under the same prompt id
        if prompt_id in known_hashes and row["prompt_sha256"] != known_hashes[prompt_id]:
            raise ValueError(f"Prompt {prompt_id} changed since it was collected; use a new output file")
        # also avoid mixing answers made under different shared instructions
        if row["system_instruction"] != SYSTEM_INSTRUCTION:
            raise ValueError("System instruction changed since collection; use a new output file")
        completed.add((prompt_id, row["model_id"]))
    return completed


def collect(client: OpenAI, prompts: list[dict[str, str]], models: list[dict[str, str]],
            output: Path, temperature: float, max_tokens: int) -> int:
    """collect one answer per prompt/model pair and checkpoint each completed row"""
    # read the checkpoint before opening the file for new rows
    completed = load_completed(output, prompts)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        # write the header only when the results file is new
        if handle.tell() == 0:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        for prompt in prompts:
            for model in models:
                key = (prompt["prompt_id"], model["model_id"])
                # skip pairs already saved in an earlier or interrupted run
                if key in completed:
                    continue
                # start with the requested token limit and increase it only for a cut-off answer
                requested_max_tokens = max_tokens
                while True:
                    response = chat_with_retries(client, model["model_id"], prompt["LLM_Input"], temperature, requested_max_tokens)
                    # retry this pair with more room capped at 4096 tokens
                    if response.choices and response.choices[0].finish_reason == "length" and requested_max_tokens < 4096:
                        requested_max_tokens = min(requested_max_tokens * 2, 4096)
                        print(f"Retrying truncated answer for {prompt['prompt_id']} / {model['model_id']} with {requested_max_tokens} tokens")
                        continue
                    text = response_text(response)
                    break
                # save the answer and the details needed to check how it was made
                usage = response.usage
                writer.writerow({
                    "LLM_name": model["model_id"],
                    "LLM_Input": prompt["LLM_Input"],
                    "LLM_output": text,
                    "prompt_id": prompt["prompt_id"],
                    "category": prompt["category"],
                    "split": prompt["split"],
                    "model_id": model["model_id"],
                    "family": model["family"],
                    "provider": model["provider"],
                    "prompt_sha256": hashlib.sha256(prompt["LLM_Input"].encode("utf-8")).hexdigest(),
                    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "temperature": temperature,
                    "max_tokens": requested_max_tokens,
                    "response_id": response.id,
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": getattr(usage, "prompt_tokens", "") if usage else "",
                    "completion_tokens": getattr(usage, "completion_tokens", "") if usage else "",
                    "system_instruction": SYSTEM_INSTRUCTION,
                })
                # flush the completed row to disk before moving to another api request
                handle.flush()
                os.fsync(handle.fileno())
                completed.add(key)
                count += 1
                print(f"Saved {prompt['prompt_id']} / {model['model_id']} ({count} new)")
    return count


def main() -> int:
    """connect the three terminal commands: models, probe, and collect"""
    # parse command-line choices so each operation can be run separately
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("models", help="List model IDs visible to your NVIDIA key")
    listing.add_argument("--contains", default="", help="Filter IDs by text")
    for name in ("probe", "collect"):
        command = sub.add_parser(name)
        command.add_argument("--models", type=Path, default=Path("models.csv"))
    collecting = sub.choices["collect"]
    collecting.add_argument("--prompts", type=Path, default=Path("data/prompts.csv"))
    collecting.add_argument("--output", type=Path, default=Path("data/generated/responses.csv"))
    collecting.add_argument("--temperature", type=float, default=0.7)
    collecting.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    try:
        # load the secret only after parsing the command and never print it
        client = client_from_env()
        available = available_ids(client)
        if args.command == "models":
            # print visible ids so i can choose real models instead of guessing names
            for model_id in sorted(available):
                if args.contains.casefold() in model_id.casefold():
                    print(model_id)
            return 0
        models = load_models(args.models)
        verify_ids(models, available)
        if args.command == "probe":
            # send one short prompt per selected model to verify chat actually works
            for model in models:
                response = chat_with_retries(client, model["model_id"], "Reply with the word OK.", 0.0, 512)
                response_text(response)
                print(f"Available for chat: {model['model_id']} ({model['family']})")
            return 0
        if not 0 <= args.temperature <= 1 or args.max_tokens < 1:
            raise ValueError("Use temperature 0..1 and max-tokens >= 1")
        # collect only after the input files and settings pass the earlier checks
        prompts = load_prompts(args.prompts)
        count = collect(client, prompts, models, args.output, args.temperature, args.max_tokens)
        print(f"Done: {count} new responses saved to {args.output}")
        return 0
    except (OSError, ValueError, APIStatusError, APIConnectionError, APITimeoutError) as exc:
        # print a useful error without dumping a request body or api key
        if isinstance(exc, APIStatusError):
            message = f"API request failed with HTTP {exc.status_code} ({type(exc).__name__})"
        elif isinstance(exc, (APIConnectionError, APITimeoutError)):
            message = f"API request failed ({type(exc).__name__})"
        else:
            message = str(exc)
        print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
