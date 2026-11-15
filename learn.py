import argparse
import os
import shutil
import sys
from datetime import datetime
from voyager import Voyager

# R2: optional model/endpoint override (defaults = paper setup)
MODEL_MAIN = os.environ.get("MODEL_MAIN", "gpt-5.4-mini")
MODEL_SMALL = os.environ.get("MODEL_SMALL", "gpt-5.4-nano")
if os.environ.get("LLM_BASE_URL"):  # local OpenAI-compatible server for chat models only; embeddings stay on OpenAI
    import langchain_openai
    _init = langchain_openai.ChatOpenAI.__init__
    def _patched(self, *a, **k):
        k.setdefault("base_url", os.environ["LLM_BASE_URL"]); k.setdefault("api_key", "local")
        _init(self, *a, **k)
    langchain_openai.ChatOpenAI.__init__ = _patched

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit(
        "Error: OPENAI_API_KEY environment variable is not set.\n"
        "Set it with: export OPENAI_API_KEY='sk-...'"
    )

parser = argparse.ArgumentParser()
parser.add_argument("-max", type=int, default=160)
parser.add_argument("-id",   type=str, default=None, help="run id suffix (e.g. exp1 → run-exp1)")
parser.add_argument("-from", dest="from_id", type=str, default=None,
                    help="source run id to copy checkpoint from (e.g. run-2026-03-31_14-05-54)")
parser.add_argument("-ablation", dest="ablation", type=str, default="none",
                    choices=["none", "no_dedup", "no_critique", "no_curriculum", "voyager_infra"],
                    help="Ablation mode: 'none' (full AutoSkill), "
                         "'no_dedup' (disable deduplication gate), "
                         "'no_critique' (disable reuse critique injection)")
parser.add_argument("-mc-port", dest="mc_port", type=int, default=25565)
parser.add_argument("-server-port", dest="server_port", type=int, default=3000)
parser.add_argument("-mc-world-path", dest="mc_world_path", type=str,
                    default=os.path.expanduser("~/mc_server_119/world"))
args = parser.parse_args()

# ── Determine run_id ──────────────────────────────────────────────────────────
if args.id:
    run_id = f"run-{args.id}"
else:
    run_id = datetime.now().strftime("run-%Y-%m-%d_%H-%M-%S")

# ── Copy checkpoint from a different run (-from) ──────────────────────────────
resume = False
fresh_world = False

if args.from_id:
    # -from: copy checkpoint from another run, then resume under new run_id
    src_ckpt = os.path.join("ckpt", args.from_id)
    dst_ckpt = os.path.join("ckpt", run_id)
    src_logs = os.path.join("logs", args.from_id)
    dst_logs = os.path.join("logs", run_id)
    if not os.path.isdir(src_ckpt):
        raise FileNotFoundError(
            f"Source checkpoint not found: {src_ckpt}\n"
            f"Available runs: {sorted(os.listdir('ckpt'))}"
        )
    if os.path.isdir(dst_ckpt):
        print(f"[learn] Checkpoint {dst_ckpt} already exists — resuming in place")
    else:
        print(f"[learn] Copying checkpoint  {src_ckpt} → {dst_ckpt}")
        shutil.copytree(src_ckpt, dst_ckpt)
    if os.path.isdir(src_logs) and not os.path.isdir(dst_logs):
        print(f"[learn] Copying logs        {src_logs} → {dst_logs}")
        shutil.copytree(src_logs, dst_logs)
    resume = True
    fresh_world = True  # different checkpoint source → hard-reset world to avoid inventory mismatch

elif os.path.isdir(os.path.join("ckpt", run_id)):
    # ckpt already exists for this run_id → always resume (even with -max)
    resume = True
    print(f"[learn] Resuming existing run {run_id}")

# ─────────────────────────────────────────────────────────────────────────────
v = Voyager(
    mc_port=args.mc_port,
    server_port=args.server_port,
    mc_world_path=args.mc_world_path,
    openai_api_key=OPENAI_API_KEY,
    max_iterations=args.max,
    run_id=run_id,
    resume=resume,
    fresh_world=fresh_world,
    env_request_timeout=180,
    action_agent_task_max_retries=2,
    action_agent_model_name=MODEL_MAIN,
    critic_agent_model_name=MODEL_MAIN,
    curriculum_agent_model_name=MODEL_MAIN,
    curriculum_agent_qa_model_name=MODEL_SMALL,
    skill_manager_model_name=MODEL_SMALL,
    ablation=args.ablation,
)
v.learn()
os._exit(0)
