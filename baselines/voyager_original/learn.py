import argparse
import os
import shutil
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

parser = argparse.ArgumentParser()
parser.add_argument("-max", type=int, default=160)
parser.add_argument("-id", type=str, default=None)
parser.add_argument("-from", dest="from_id", type=str, default=None,
                    help="source run id to copy checkpoint from")
args = parser.parse_args()

if args.id:
    run_id = f"run-{args.id}"
else:
    run_id = datetime.now().strftime("run-%Y-%m-%d_%H-%M-%S")

resume = False
if args.from_id:
    src = os.path.join("ckpt", f"run-{args.from_id}")
    dst = os.path.join("ckpt", run_id)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Source checkpoint not found: {src}")
    if os.path.isdir(dst):
        print(f"[learn] Checkpoint {dst} already exists — resuming in place")
    else:
        print(f"[learn] Copying checkpoint  {src} → {dst}")
        shutil.copytree(src, dst)
    resume = True
elif os.path.isdir(os.path.join("ckpt", run_id)):
    resume = True
    print(f"[learn] Resuming existing run {run_id}")

v = Voyager(
    mc_port=int(os.environ.get("MC_PORT", 25565)),
    server_port=int(os.environ.get("SERVER_PORT", 3000)),
    mc_world_path=os.environ.get("MC_WORLD", os.path.expanduser("~/mc_server_119/world")),
    openai_api_key=os.environ["OPENAI_API_KEY"],
    max_iterations=args.max,
    run_id=run_id,
    resume=resume,
    action_agent_task_max_retries=2,
    env_request_timeout=180,
    action_agent_model_name=MODEL_MAIN,
    critic_agent_model_name=MODEL_MAIN,
    curriculum_agent_model_name=MODEL_MAIN,
    curriculum_agent_qa_model_name=MODEL_SMALL,
    skill_manager_model_name=MODEL_SMALL,
)
v.learn()
os._exit(0)
