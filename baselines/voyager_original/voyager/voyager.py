import copy
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict

import voyager.utils as U
from .env import VoyagerEnv

from .agents import ActionAgent
from .agents import CriticAgent
from .agents import CurriculumAgent
from .agents import SkillManager


# TODO: remove event memory
class Voyager:
    def __init__(
        self,
        mc_port: int = None,
        azure_login: Dict[str, str] = None,
        server_port: int = 3000,
        offline_mode: bool = False,
        offline_username: str = "Player",
        mc_world_path: str = None,
        openai_api_key: str = None,
        env_wait_ticks: int = 20,
        env_request_timeout: int = 120,
        max_iterations: int = 160,
        reset_placed_if_failed: bool = False,
        action_agent_model_name: str = "gpt-4",
        action_agent_temperature: float = 0,
        action_agent_task_max_retries: int = 4,
        action_agent_show_chat_log: bool = True,
        action_agent_show_execution_error: bool = True,
        curriculum_agent_model_name: str = "gpt-4",
        curriculum_agent_temperature: float = 0,
        curriculum_agent_qa_model_name: str = "gpt-3.5-turbo",
        curriculum_agent_qa_temperature: float = 0,
        curriculum_agent_warm_up: Dict[str, int] = None,
        curriculum_agent_core_inventory_items: str = r".*_log|.*_planks|stick|crafting_table|furnace"
        r"|cobblestone|dirt|coal|.*_pickaxe|.*_sword|.*_axe",
        curriculum_agent_mode: str = "auto",
        critic_agent_model_name: str = "gpt-4",
        critic_agent_temperature: float = 0,
        critic_agent_mode: str = "auto",
        skill_manager_model_name: str = "gpt-3.5-turbo",
        skill_manager_temperature: float = 0,
        skill_manager_retrieval_top_k: int = 5,
        openai_api_request_timeout: int = 240,
        ckpt_dir: str = "ckpt",
        skill_library_dir: str = None,
        resume: bool = False,
        run_id: str = None,
    ):
        """
        The main class for Voyager.
        Action agent is the iterative prompting mechanism in paper.
        Curriculum agent is the automatic curriculum in paper.
        Critic agent is the self-verification in paper.
        Skill manager is the skill library in paper.
        :param mc_port: minecraft in-game port
        :param azure_login: minecraft login config
        :param server_port: mineflayer port
        :param openai_api_key: openai api key
        :param env_wait_ticks: how many ticks at the end each step will wait, if you found some chat log missing,
        you should increase this value
        :param env_request_timeout: how many seconds to wait for each step, if the code execution exceeds this time,
        python side will terminate the connection and need to be resumed
        :param reset_placed_if_failed: whether to reset placed blocks if failed, useful for building task
        :param action_agent_model_name: action agent model name
        :param action_agent_temperature: action agent temperature
        :param action_agent_task_max_retries: how many times to retry if failed
        :param curriculum_agent_model_name: curriculum agent model name
        :param curriculum_agent_temperature: curriculum agent temperature
        :param curriculum_agent_qa_model_name: curriculum agent qa model name
        :param curriculum_agent_qa_temperature: curriculum agent qa temperature
        :param curriculum_agent_warm_up: info will show in curriculum human message
        if completed task larger than the value in dict, available keys are:
        {
            "context": int,
            "biome": int,
            "time": int,
            "other_blocks": int,
            "nearby_entities": int,
            "health": int,
            "hunger": int,
            "position": int,
            "equipment": int,
            "chests": int,
            "optional_inventory_items": int,
        }
        :param curriculum_agent_core_inventory_items: only show these items in inventory before optional_inventory_items
        reached in warm up
        :param curriculum_agent_mode: "auto" for automatic curriculum, "manual" for human curriculum
        :param critic_agent_model_name: critic agent model name
        :param critic_agent_temperature: critic agent temperature
        :param critic_agent_mode: "auto" for automatic critic ,"manual" for human critic
        :param skill_manager_model_name: skill manager model name
        :param skill_manager_temperature: skill manager temperature
        :param skill_manager_retrieval_top_k: how many skills to retrieve for each task
        :param openai_api_request_timeout: how many seconds to wait for openai api
        :param ckpt_dir: checkpoint dir
        :param skill_library_dir: skill library dir
        :param resume: whether to resume from checkpoint
        """
        # set up run id — reuse if given (resume), otherwise create fresh
        if run_id is not None:
            self.run_id = run_id
        else:
            self.run_id = datetime.now().strftime("run-%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join("logs", self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        run_ckpt_dir = os.path.join(ckpt_dir, self.run_id)
        self.run_ckpt_dir = run_ckpt_dir

        # store for meta.json
        self._meta = {
            "action_model": action_agent_model_name,
            "critic_model": critic_agent_model_name,
            "skill_model": skill_manager_model_name,
            "curriculum_model": curriculum_agent_model_name,
            "embedding_model": "text-embedding-3-small",
            "max_iterations": max_iterations,
            "env_wait_ticks": env_wait_ticks,
            "env_request_timeout": env_request_timeout,
            "openai_api_request_timeout": openai_api_request_timeout,
            "action_agent_task_max_retries": action_agent_task_max_retries,
            "curriculum_mode": curriculum_agent_mode,
            "critic_mode": critic_agent_mode,
            "skill_retrieval_top_k": skill_manager_retrieval_top_k,
        }

        # tee Python stdout/stderr → python.log (resume 시 이어붙이기)
        if not isinstance(sys.stdout, U.Tee):
            _log_file = open(
                os.path.join(self.log_dir, "python.log"), "a", encoding="utf-8", buffering=1
            )
            sys.stdout = U.Tee(sys.__stdout__, _log_file)
            sys.stderr = U.Tee(sys.__stderr__, _log_file)
        print(f"[Voyager] run_id: {self.run_id}")

        # init env
        self.env = VoyagerEnv(
            mc_port=mc_port,
            azure_login=azure_login,
            server_port=server_port,
            request_timeout=env_request_timeout,
            offline_mode=offline_mode,
            offline_username=offline_username,
            log_path=self.log_dir,
            mc_world_path=mc_world_path,
        )
        self.env_wait_ticks = env_wait_ticks
        self.reset_placed_if_failed = reset_placed_if_failed
        self.max_iterations = max_iterations

        # set openai api key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # init agents
        self.action_agent = ActionAgent(
            model_name=action_agent_model_name,
            temperature=action_agent_temperature,
            request_timout=openai_api_request_timeout,
            ckpt_dir=run_ckpt_dir,
            resume=resume,
            chat_log=action_agent_show_chat_log,
            execution_error=action_agent_show_execution_error,
            log_dir=self.log_dir,
        )
        self.action_agent_task_max_retries = action_agent_task_max_retries
        self.curriculum_agent = CurriculumAgent(
            model_name=curriculum_agent_model_name,
            temperature=curriculum_agent_temperature,
            qa_model_name=curriculum_agent_qa_model_name,
            qa_temperature=curriculum_agent_qa_temperature,
            request_timout=openai_api_request_timeout,
            ckpt_dir=run_ckpt_dir,
            resume=resume,
            mode=curriculum_agent_mode,
            warm_up=curriculum_agent_warm_up,
            core_inventory_items=curriculum_agent_core_inventory_items,
            log_dir=self.log_dir,
        )
        self.critic_agent = CriticAgent(
            model_name=critic_agent_model_name,
            temperature=critic_agent_temperature,
            request_timout=openai_api_request_timeout,
            mode=critic_agent_mode,
            log_dir=self.log_dir,
        )
        self.skill_manager = SkillManager(
            model_name=skill_manager_model_name,
            temperature=skill_manager_temperature,
            retrieval_top_k=skill_manager_retrieval_top_k,
            request_timout=openai_api_request_timeout,
            ckpt_dir=skill_library_dir if skill_library_dir else run_ckpt_dir,
            resume=True if resume or skill_library_dir else False,
            log_dir=self.log_dir,
        )
        self.recorder = U.EventRecorder(ckpt_dir=run_ckpt_dir, resume=resume)
        self.resume = resume

        # init variables for rollout
        self.action_agent_rollout_num_iter = -1
        self.task = None
        self.context = ""
        self.messages = None
        self.conversations = []
        self.last_events = None
        # cumulative distinct items ever observed — persisted in checkpoint
        seen_items_path = os.path.join(self.run_ckpt_dir, "seen_items.json")
        if resume and os.path.exists(seen_items_path):
            with open(seen_items_path) as f:
                self.seen_items: set = set(json.load(f))
            print(f"[Voyager] Loaded {len(self.seen_items)} seen items from checkpoint")
        else:
            self.seen_items: set = set()

    def _update_seen_items(self, events) -> int:
        """Grow the cumulative seen_items set, persist it, and return its size."""
        if events:
            obs = events[-1][1]
            self.seen_items.update(obs.get("inventory", {}).keys())
            for slot in obs.get("status", {}).get("equipment", []):
                if slot:
                    self.seen_items.add(slot)
        for chest_contents in self.action_agent.chest_memory.values():
            if isinstance(chest_contents, dict):
                self.seen_items.update(chest_contents.keys())
        seen_items_path = os.path.join(self.run_ckpt_dir, "seen_items.json")
        with open(seen_items_path, "w") as f:
            json.dump(list(self.seen_items), f)
        return len(self.seen_items)

    def _write_meta(self):
        meta_path = os.path.join(self.log_dir, "meta.json")
        if os.path.exists(meta_path):
            return
        meta = dict(self._meta)
        import subprocess
        try:
            meta["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
            ).decode().strip()
        except Exception:
            meta["git_commit"] = None
        try:
            meta["git_branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=os.path.dirname(__file__)
            ).decode().strip()
        except Exception:
            meta["git_branch"] = None
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def _compute_tech_tree_score(items) -> int:
        """Compute Tech Tree Depth Score for a collection of item names."""
        try:
            from tech_tree_score import tech_tree_score as _tts
            return _tts(list(items))
        except Exception:
            pass
        # Inline minimal fallback (tier 1-6)
        TIERS = {
            # T2
            "oak_planks":2,"birch_planks":2,"spruce_planks":2,"stick":2,"charcoal":2,
            "coal":2,"coal_ore":2,"raw_iron":2,"iron_ore":2,"raw_copper":2,"copper_ore":2,
            "raw_gold":2,"gold_ore":2,"diamond":2,"diamond_ore":2,"emerald":2,
            "lapis_lazuli":2,"redstone":2,"ancient_debris":2,"wheat":2,"bread":2,
            "torch":2,"crafting_table":2,"furnace":2,
            # T3
            "wooden_pickaxe":3,"wooden_axe":3,"wooden_sword":3,"stone_pickaxe":3,
            "stone_axe":3,"stone_sword":3,"iron_ingot":3,"copper_ingot":3,
            "gold_ingot":3,"bucket":3,"chest":3,"glass":3,"shears":3,
            # T4
            "iron_pickaxe":4,"iron_axe":4,"iron_sword":4,"iron_helmet":4,
            "iron_chestplate":4,"iron_leggings":4,"iron_boots":4,"shield":4,"anvil":4,
            # T5
            "diamond_pickaxe":5,"diamond_axe":5,"diamond_sword":5,"diamond_helmet":5,
            "diamond_chestplate":5,"diamond_leggings":5,"diamond_boots":5,"enchanting_table":5,
            # T6
            "netherite_ingot":6,"netherite_pickaxe":6,"netherite_sword":6,"beacon":6,
        }
        return sum(TIERS.get(i.lower(), 1) for i in items)

    def _append_jsonl(self, filename: str, payload: dict):
        record = dict(payload)
        record.setdefault("ts", time.time())
        with open(os.path.join(self.log_dir, filename), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _defined_function_names(self, code_text: str):
        return sorted(
            set(
                re.findall(
                    r"async function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*bot\s*\)",
                    code_text,
                )
            )
        )

    def _classify_reuse(self, true_called, redefined):
        if redefined:
            return "redefined_known"
        if true_called:
            return "called_existing"
        return "no_match"

    def _called_skill_names(self, code_text: str):
        names = []
        for name in self.skill_manager.skills.keys():
            pattern = rf"\bawait\s+{re.escape(name)}\s*\(\s*bot\b"
            if re.search(pattern, code_text):
                names.append(name)
        return sorted(set(names))

    def reset(self, task, context="", reset_env=True):
        self.action_agent_rollout_num_iter = 0
        self.task = task
        self.context = context
        if reset_env:
            self.env.reset(
                options={
                    "mode": "soft",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
        difficulty = (
            "easy" if len(self.curriculum_agent.completed_tasks) > 15 else "peaceful"
        )
        # step to peek an observation
        events = self.env.step(
            "bot.chat(`/time set ${getNextTime()}`);\n"
            + f"bot.chat('/difficulty {difficulty}');"
        )
        skills = self.skill_manager.retrieve_skills(query=self.context)
        print(
            f"\033[33mRender Action Agent system message with {len(skills)} skills\033[0m"
        )
        system_message = self.action_agent.render_system_message(skills=skills)
        human_message = self.action_agent.render_human_message(
            events=events, code="", task=self.task, context=context, critique=""
        )
        self.messages = [system_message, human_message]
        print(
            f"\033[32m****Action Agent human message****\n{human_message.content}\033[0m"
        )
        assert len(self.messages) == 2
        self.conversations = []
        return self.messages

    def close(self):
        self.env.close()

    def step(self):
        t0 = time.perf_counter()
        error = None
        if self.action_agent_rollout_num_iter < 0:
            raise ValueError("Agent must be reset before stepping")
        ai_message = self.action_agent.llm(self.messages)
        print(f"\033[34m****Action Agent ai message****\n{ai_message.content}\033[0m")
        self.conversations.append(
            (self.messages[0].content, self.messages[1].content, ai_message.content)
        )
        parsed_result = self.action_agent.process_ai_message(message=ai_message)
        success = False
        if isinstance(parsed_result, dict):
            code = parsed_result["program_code"] + "\n" + parsed_result["exec_code"]
            defined_names = set(self._defined_function_names(parsed_result["program_code"]))
            await_called = set(self._called_skill_names(code))
            true_called = sorted(name for name in await_called if name not in defined_names)
            redefined_known = sorted(name for name in defined_names if name in self.skill_manager.skills)
            reuse_mode = self._classify_reuse(true_called, redefined_known)
            self._append_jsonl("reuse_log.jsonl", {
                "task": self.task,
                "effective_task": self.task,
                "exact_skill": None,
                "family_skill": None,
                "exact_skill_atomic": None,
                "family_skill_atomic": None,
                "retrieved_skill_names": list(self.skill_manager.last_retrieved_names),
                "defined_function_names": sorted(defined_names),
                "await_called_known_skills": sorted(await_called),
                "true_called_known_skills": true_called,
                "redefined_known_skills": redefined_known,
                "reuse_mode": reuse_mode,
                "code_preview": code[:500],
            })
            events = self.env.step(
                code,
                programs=self.skill_manager.programs,
            )
            self.recorder.record(events, self.task)
            self.action_agent.update_chest_memory(events[-1][1]["nearbyChests"])
            success, critique = self.critic_agent.check_task_success(
                events=events,
                task=self.task,
                context=self.context,
                chest_observation=self.action_agent.render_chest_observation(),
                max_retries=5,
            )

            if self.reset_placed_if_failed and not success:
                # revert all the placing event in the last step
                blocks = []
                positions = []
                for event_type, event in events:
                    if event_type == "onSave" and event["onSave"].endswith("_placed"):
                        block = event["onSave"].split("_placed")[0]
                        position = event["status"]["position"]
                        blocks.append(block)
                        positions.append(position)
                new_events = self.env.step(
                    f"await givePlacedItemBack(bot, {U.json_dumps(blocks)}, {U.json_dumps(positions)})",
                    programs=self.skill_manager.programs,
                )
                events[-1][1]["inventory"] = new_events[-1][1]["inventory"]
                events[-1][1]["voxels"] = new_events[-1][1]["voxels"]
            new_skills = self.skill_manager.retrieve_skills(
                query=self.context
                + "\n\n"
                + self.action_agent.summarize_chatlog(events)
            )
            system_message = self.action_agent.render_system_message(skills=new_skills)
            human_message = self.action_agent.render_human_message(
                events=events,
                code=parsed_result["program_code"],
                task=self.task,
                context=self.context,
                critique=critique,
            )
            self.last_events = copy.deepcopy(events)
            self.messages = [system_message, human_message]
        else:
            assert isinstance(parsed_result, str)
            self.recorder.record([], self.task)
            print(f"\033[34m{parsed_result} Trying again!\033[0m")
        assert len(self.messages) == 2
        self.action_agent_rollout_num_iter += 1
        done = (
            self.action_agent_rollout_num_iter >= self.action_agent_task_max_retries
            or success
        )
        info = {
            "task": self.task,
            "success": success,
            "conversations": self.conversations,
        }
        if success:
            assert (
                "program_code" in parsed_result and "program_name" in parsed_result
            ), "program and program_name must be returned when success"
            info["program_code"] = parsed_result["program_code"]
            info["program_name"] = parsed_result["program_name"]
        else:
            print(
                f"\033[32m****Action Agent human message****\n{self.messages[-1].content}\033[0m"
            )
        _distinct = self._update_seen_items(self.last_events)
        self._append_jsonl("step_log.jsonl", {
            "task": self.task,
            "step": self.action_agent_rollout_num_iter,
            "success": success,
            "tech_tree_score": self._compute_tech_tree_score(self.seen_items),
            "distinct_items": _distinct,
            "seen_items": sorted(self.seen_items),
            "agent_iterations": self.action_agent_rollout_num_iter,
            "duration_s": time.perf_counter() - t0,
            "error": error,
            "skill_library_size": len(self.skill_manager.skills),
        })
        return self.messages, 0, done, info

    def rollout(self, *, task, context, reset_env=True):
        self.reset(task=task, context=context, reset_env=reset_env)
        while True:
            messages, reward, done, info = self.step()
            if done:
                break
        return messages, reward, done, info

    def learn(self, reset_env=True):
        self._write_meta()
        if self.resume:
            # keep the inventory
            self.env.reset(
                options={
                    "mode": "soft",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
        else:
            # clear the inventory
            self.env.reset(
                options={
                    "mode": "hard",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
            self.resume = True
        self.last_events = self.env.step("")

        while True:
            if self.recorder.iteration > self.max_iterations:
                print("Iteration limit reached")
                break
            task, context = self.curriculum_agent.propose_next_task(
                events=self.last_events,
                chest_observation=self.action_agent.render_chest_observation(),
                max_retries=5,
            )
            print(
                f"\033[35mStarting task {task} for at most {self.action_agent_task_max_retries} times\033[0m"
            )
            try:
                messages, reward, done, info = self.rollout(
                    task=task,
                    context=context,
                    reset_env=reset_env,
                )
            except Exception as e:
                time.sleep(3)  # wait for mineflayer to exit
                info = {
                    "task": task,
                    "success": False,
                }
                # reset bot status — no position tp, bot always spawns at
                # setworldspawn (spawnRadius 0 gamerule keeps it deterministic)
                self.last_events = self.env.reset(
                    options={
                        "mode": "hard",
                        "wait_ticks": self.env_wait_ticks,
                        "inventory": self.last_events[-1][1]["inventory"],
                        "equipment": self.last_events[-1][1]["status"]["equipment"],
                    }
                )
                # use red color background to print the error
                print("Your last round rollout terminated due to error:")
                print(f"\033[41m{e}\033[0m")

            if info["success"]:
                self.skill_manager.add_new_skill(info)
                self._append_jsonl("skill_log.jsonl", {
                    "kind": "skill_add",
                    "task": info["task"],
                    "program_name": info["program_name"],
                    "family_key": None,
                    "parse_mode": None,
                    "nearest_score": None,
                    "matched_neighbor": None,
                    "canonical_desc": None,
                    "dedup_group_id": None,
                    "requested_count": None,
                    "requested_target": None,
                    "original_task": info["task"],
                    "effective_task": info["task"],
                    "atomic_mode": False,
                    "atomic_repeat_count": 1,
                })

            self.curriculum_agent.update_exploration_progress(info)
            print(
                f"\033[35mCompleted tasks: {', '.join(self.curriculum_agent.completed_tasks)}\033[0m"
            )
            print(
                f"\033[35mFailed tasks: {', '.join(self.curriculum_agent.failed_tasks)}\033[0m"
            )

        return {
            "completed_tasks": self.curriculum_agent.completed_tasks,
            "failed_tasks": self.curriculum_agent.failed_tasks,
            "skills": self.skill_manager.skills,
        }

    def decompose_task(self, task):
        if not self.last_events:
            self.last_events = self.env.reset(
                options={
                    "mode": "hard",
                    "wait_ticks": self.env_wait_ticks,
                }
            )
        return self.curriculum_agent.decompose_task(task, self.last_events)

    def inference(self, task=None, sub_goals=[], reset_mode="hard", reset_env=True):
        if not task and not sub_goals:
            raise ValueError("Either task or sub_goals must be provided")
        if not sub_goals:
            sub_goals = self.decompose_task(task)
        self.env.reset(
            options={
                "mode": reset_mode,
                "wait_ticks": self.env_wait_ticks,
            }
        )
        self.curriculum_agent.completed_tasks = []
        self.curriculum_agent.failed_tasks = []
        self.last_events = self.env.step("")
        while self.curriculum_agent.progress < len(sub_goals):
            next_task = sub_goals[self.curriculum_agent.progress]
            context = self.curriculum_agent.get_task_context(next_task)
            print(
                f"\033[35mStarting task {next_task} for at most {self.action_agent_task_max_retries} times\033[0m"
            )
            messages, reward, done, info = self.rollout(
                task=next_task,
                context=context,
                reset_env=reset_env,
            )
            self.curriculum_agent.update_exploration_progress(info)
            print(
                f"\033[35mCompleted tasks: {', '.join(self.curriculum_agent.completed_tasks)}\033[0m"
            )
            print(
                f"\033[35mFailed tasks: {', '.join(self.curriculum_agent.failed_tasks)}\033[0m"
            )
