from __future__ import annotations

import os
import random
import re
from collections import Counter

import voyager.utils as U
from voyager.prompts import load_prompt
from voyager.utils.json_utils import fix_and_parse_json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_chroma import Chroma


class CurriculumAgent:
    def __init__(
        self,
        model_name="gpt-3.5-turbo",
        temperature=0,
        qa_model_name="gpt-3.5-turbo",
        qa_temperature=0,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        mode="auto",
        warm_up=None,
        core_inventory_items: str | None = None,
        log_dir=None,
    ):
        callbacks = []
        qa_callbacks = []
        if log_dir is not None:
            token_logger = U.TokenLogger(
                agent_name="curriculum",
                log_path=os.path.join(log_dir, "token_log.jsonl"),
            )
            token_logger_qa = U.TokenLogger(
                agent_name="curriculum_qa",
                log_path=os.path.join(log_dir, "token_log.jsonl"),
            )
            callbacks.append(token_logger)
            qa_callbacks.append(token_logger_qa)
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
            max_tokens=512,
            callbacks=callbacks,
        )
        self.qa_llm = ChatOpenAI(
            model_name=qa_model_name,
            temperature=qa_temperature,
            request_timeout=request_timout,
            callbacks=qa_callbacks,
        )
        assert mode in [
            "auto",
            "manual",
        ], f"mode {mode} not supported"
        self.mode = mode
        self.ckpt_dir = ckpt_dir
        U.f_mkdir(f"{ckpt_dir}/curriculum/vectordb")
        if resume:
            print(f"\033[35mLoading Curriculum Agent from {ckpt_dir}/curriculum\033[0m")
            self.completed_tasks = U.load_json(
                f"{ckpt_dir}/curriculum/completed_tasks.json"
            )
            self.failed_tasks = U.load_json(f"{ckpt_dir}/curriculum/failed_tasks.json")
            self.qa_cache = U.load_json(f"{ckpt_dir}/curriculum/qa_cache.json")
        else:
            self.completed_tasks = []
            self.failed_tasks = []
            self.qa_cache = {}
        # track consecutive deposit failures per chest position
        self.failed_deposit_chests = {}
        # vectordb for qa cache
        self.qa_cache_questions_vectordb = Chroma(
            collection_name="qa_cache_questions_vectordb",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{ckpt_dir}/curriculum/vectordb",
        )
        assert self.qa_cache_questions_vectordb._collection.count() == len(
            self.qa_cache
        ), (
            f"Curriculum Agent's qa cache question vectordb is not synced with qa_cache.json.\n"
            f"There are {self.qa_cache_questions_vectordb._collection.count()} questions in vectordb "
            f"but {len(self.qa_cache)} questions in qa_cache.json.\n"
            f"Did you set resume=False when initializing the agent?\n"
            f"You may need to manually delete the qa cache question vectordb directory for running from scratch.\n"
        )
        # if warm up not defined, initialize it as a dict, else, initialize all the missing value as a default value
        if not warm_up:
            warm_up = self.default_warmup
        self.warm_up = {}
        if "optional_inventory_items" in warm_up:
            assert core_inventory_items is not None
            self._core_inv_items_regex = re.compile(core_inventory_items)
            self.warm_up["optional_inventory_items"] = warm_up[
                "optional_inventory_items"
            ]
        else:
            self.warm_up["optional_inventory_items"] = 0
        for key in self.curriculum_observations:
            self.warm_up[key] = warm_up.get(key, self.default_warmup[key])
        self.warm_up["nearby_blocks"] = 0
        self.warm_up["inventory"] = 0
        self.warm_up["completed_tasks"] = 0
        self.warm_up["failed_tasks"] = 0

    @property
    def default_warmup(self):
        return {
            "context": 15,
            "biome": 10,
            "time": 15,
            "nearby_blocks": 0,
            "other_blocks": 10,
            "nearby_entities": 5,
            "health": 15,
            "hunger": 15,
            "position": 0,
            "equipment": 0,
            "inventory": 0,
            "optional_inventory_items": 7,
            "chests": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
        }

    @property
    def curriculum_observations(self):
        return [
            "context",
            "biome",
            "time",
            "nearby_blocks",
            "other_blocks",
            "nearby_entities",
            "health",
            "hunger",
            "position",
            "equipment",
            "inventory",
            "chests",
            "completed_tasks",
            "failed_tasks",
        ]

    @property
    def progress(self):
        return len(self.completed_tasks)

    def render_system_message(self):
        system_message = SystemMessage(content=load_prompt(getattr(self, "prompt_name", "curriculum")))
        assert isinstance(system_message, SystemMessage)
        return system_message

    def render_observation(self, *, events, chest_observation):
        assert events[-1][0] == "observe", "Last event must be observe"
        event = events[-1][1]
        biome = event["status"]["biome"]
        time_of_day = event["status"]["timeOfDay"]
        voxels = event["voxels"]
        block_records = event["blockRecords"]
        entities = event["status"]["entities"]
        health = event["status"]["health"]
        hunger = event["status"]["food"]
        position = event["status"]["position"]
        equipment = event["status"]["equipment"]
        inventory_used = event["status"]["inventoryUsed"]
        inventory = event["inventory"]

        if not any(
            "dirt" in block
            or "log" in block
            or "grass" in block
            or "sand" in block
            or "snow" in block
            for block in voxels
        ):
            biome = "underground"

        other_blocks = ", ".join(
            list(
                set(block_records).difference(set(voxels).union(set(inventory.keys())))
            )
        )

        other_blocks = other_blocks if other_blocks else "None"

        nearby_entities = (
            ", ".join([k for k, v in sorted(entities.items(), key=lambda x: x[1])])
            if entities
            else "None"
        )

        # Build completed_tasks string with counts; mark repeats clearly
        if self.completed_tasks:
            task_counts = Counter(self.completed_tasks)
            parts = []
            seen = set()
            for t in self.completed_tasks:
                if t not in seen:
                    seen.add(t)
                    n = task_counts[t]
                    parts.append(f"{t} (x{n})" if n > 1 else t)
            completed_tasks = ", ".join(parts)
        else:
            completed_tasks = "None"
        if self.failed_tasks:
            fail_counts = Counter(self.failed_tasks)
            fail_parts = []
            seen_f = set()
            for t in self.failed_tasks:
                if t not in seen_f:
                    seen_f.add(t)
                    n = fail_counts[t]
                    fail_parts.append(f"{t} (x{n})" if n > 1 else t)
            failed_tasks = ", ".join(fail_parts)
        else:
            failed_tasks = "None"

        # filter out optional inventory items if required
        if self.progress < self.warm_up["optional_inventory_items"]:
            inventory = {
                k: v
                for k, v in inventory.items()
                if self._core_inv_items_regex.search(k) is not None
            }

        observation = {
            "context": "",
            "biome": f"Biome: {biome}\n\n",
            "time": f"Time: {time_of_day}\n\n",
            "nearby_blocks": f"Nearby blocks: {', '.join(voxels) if voxels else 'None'}\n\n",
            "other_blocks": f"Other blocks that are recently seen: {other_blocks}\n\n",
            "nearby_entities": f"Nearby entities: {nearby_entities}\n\n",
            "health": f"Health: {health:.1f}/20\n\n",
            "hunger": f"Hunger: {hunger:.1f}/20\n\n",
            "position": f"Position: x={position['x']:.1f}, y={position['y']:.1f}, z={position['z']:.1f}\n\n",
            "equipment": f"Equipment: {equipment}\n\n",
            "inventory": f"Inventory ({inventory_used}/36): {inventory if inventory else 'Empty'}\n\n",
            "chests": chest_observation,
            "completed_tasks": f"Completed tasks so far: {completed_tasks}\n(Tasks marked x2 or more have already been mastered — do not propose them again unless required as a prerequisite for a harder task.)\nRecent task history (last 5): {', '.join(self.completed_tasks[-5:]) if self.completed_tasks else 'None'} — do NOT repeat any of these.\n\n",
            "failed_tasks": f"Failed tasks that are too hard: {failed_tasks}\n(Tasks marked x2 or more have been attempted multiple times and MUST NOT be proposed again — they are permanently blocked.)\n\n",
        }
        return observation

    def render_human_message(self, *, events, chest_observation,
                              skill_library=None, tier_summary=None):
        content = ""
        observation = self.render_observation(
            events=events, chest_observation=chest_observation
        )
        if self.progress >= self.warm_up["context"]:
            questions, answers = self.run_qa(
                events=events, chest_observation=chest_observation
            )
            i = 1
            for question, answer in zip(questions, answers):
                if "Answer: Unknown" in answer or "language model" in answer:
                    continue
                observation["context"] += f"Question {i}: {question}\n"
                observation["context"] += f"{answer}\n\n"
                i += 1
                if i > 5:
                    break

        for key in self.curriculum_observations:
            if self.progress >= self.warm_up[key]:
                if self.warm_up[key] != 0:
                    should_include = random.random() < 0.8
                else:
                    should_include = True
                if should_include:
                    content += observation[key]

        # [AutoSkill-v2][B] Append tier breakdown + library skill list
        if tier_summary:
            content += f"Tech tree progress (current inventory by tier):\n{tier_summary}\n\n"
        if skill_library:
            skill_str = ", ".join(skill_library[:40])
            suffix = f" ... ({len(skill_library)} total)" if len(skill_library) > 40 else ""
            content += (
                f"Skill library ({len(skill_library)} skills): {skill_str}{suffix}\n"
                f"→ Prefer tasks that build on or combine these existing skills.\n\n"
            )

        print(f"\033[35m****Curriculum Agent human message****\n{content}\033[0m")
        return HumanMessage(content=content)

    def propose_next_task(self, *, events, chest_observation,
                          skill_library=None, tier_summary=None, max_retries=5):
        if self.progress == 0 and self.mode == "auto":
            task = "Mine 1 wood log"
            context = "You can mine one of oak, birch, spruce, jungle, acacia, dark oak, or mangrove logs."
            return task, context

        # hard code task when inventory is almost full
        inventoryUsed = events[-1][1]["status"]["inventoryUsed"]
        if inventoryUsed >= 33:
            if chest_observation != "Chests: None\n\n":
                chests = chest_observation[8:-2].split("\n")
                for chest in chests:
                    content = chest.split(":")[1]
                    if content == " Unknown items inside" or content == " Empty":
                        position = chest.split(":")[0]
                        # skip chests that have failed to deposit 2+ times
                        if self.failed_deposit_chests.get(position, 0) >= 2:
                            continue
                        task = f"Deposit useless items into the chest at {position}"
                        context = (
                            f"Your inventory have {inventoryUsed} occupied slots before depositing. "
                            "After depositing, your inventory should only have 20 occupied slots. "
                            "You should deposit useless items such as andesite, dirt, cobblestone, etc. "
                            "Also, you can deposit low-level tools, "
                            "For example, if you have a stone pickaxe, you can deposit a wooden pickaxe. "
                            "Make sure the list of useless items are in your inventory "
                            "(do not list items already in the chest), "
                            "You can use bot.inventoryUsed() to check how many inventory slots are used."
                        )
                        return task, context
            # Only force chest placement if it hasn't been repeated too many times.
            # If "Place a chest" / "Craft 1 chest" have been done >= 3 times each,
            # fall through to the LLM so it can suggest smelt/craft/mine instead.
            task_counts = Counter(self.completed_tasks)
            chest_cycles = task_counts.get("Place a chest", 0) + task_counts.get(
                "Craft 1 chest", 0
            )
            # Also stop forcing chest tasks once they keep FAILING. Otherwise a
            # full inventory plus an unachievable "Craft 1 chest" traps the agent
            # in an infinite forced-chest loop here (this hard-coded path returns
            # before propose_next_ai_task, so the failed-task filter never sees
            # it), wasting every remaining iteration. When chest tasks have failed
            # repeatedly, fall through to the LLM so it can deposit/craft/mine
            # something achievable instead.
            fail_counts = Counter(self.failed_tasks)
            chest_fails = fail_counts.get("Place a chest", 0) + fail_counts.get(
                "Craft 1 chest", 0
            )
            if chest_cycles < 3 and chest_fails < 3:
                if "chest" in events[-1][1]["inventory"]:
                    task = "Place a chest"
                    context = (
                        f"You have a chest in inventory, place it around you. "
                        f"If chests is not None, or nearby blocks contains chest, this task is success."
                    )
                else:
                    task = "Craft 1 chest"
                    context = "Craft 1 chest with 8 planks of any kind of wood."
                return task, context
            # else: fall through to LLM with full inventory context

        messages = [
            self.render_system_message(),
            self.render_human_message(
                events=events, chest_observation=chest_observation,
                skill_library=skill_library, tier_summary=tier_summary,
            ),
        ]

        if self.mode == "auto":
            return self.propose_next_ai_task(messages=messages, max_retries=max_retries)
        elif self.mode == "manual":
            return self.propose_next_manual_task()
        else:
            raise ValueError(f"Invalid curriculum agent mode: {self.mode}")

    def propose_next_ai_task(self, *, messages, max_retries=5):
        if max_retries == 0:
            raise RuntimeError("Max retries reached, failed to propose ai task.")
        curriculum = self.llm(messages).content
        print(f"\033[31m****Curriculum Agent ai message****\n{curriculum}\033[0m")
        try:
            response = self.parse_ai_message(curriculum)
            assert "next_task" in response
            task = response["next_task"]

            # Hard filter 1: reject tasks already completed >= 3 times
            task_counts = Counter(self.completed_tasks)
            repeat_count = task_counts.get(task, 0)
            if repeat_count >= 3:
                over_repeated = [t for t, n in task_counts.items() if n >= 3]
                print(
                    f"\033[35mRejecting repeated task '{task}' "
                    f"(completed {repeat_count}x). Forcing retry.\033[0m"
                )
                avoidance_note = HumanMessage(
                    content=(
                        f"IMPORTANT: The task '{task}' has already been completed "
                        f"{repeat_count} times and is fully mastered. "
                        f"Do NOT propose it again. "
                        f"Tasks to avoid (mastered 3+ times): {over_repeated}. "
                        f"Propose a completely different, more advanced task."
                    )
                )
                return self.propose_next_ai_task(
                    messages=messages + [avoidance_note],
                    max_retries=max_retries - 1,
                )

            # Hard filter 2: reject tasks failed >= 3 times (currently unachievable).
            # Normalize task names before counting so that quantity/formatting
            # variants ("Smelt 9 raw_iron", "Smelt 4 raw_iron", "smelt raw_iron")
            # are treated as the same task and cannot bypass the filter by changing
            # only the number or underscore style.
            def _norm_task(t: str) -> str:
                t = re.sub(
                    r'^(craft|smelt|mine|kill|cook|equip)\s+\d+\s+',
                    lambda m: m.group(1) + ' ',
                    t, flags=re.IGNORECASE,
                )
                return t.lower().replace('_', ' ').strip()

            fail_counts_norm = Counter(_norm_task(t) for t in self.failed_tasks)
            norm_task = _norm_task(task)
            fail_count = fail_counts_norm.get(norm_task, 0)
            if fail_count >= 3:
                over_failed = [t for t, n in fail_counts_norm.items() if n >= 3]
                print(
                    f"\033[35mRejecting repeatedly-failed task '{task}' "
                    f"(norm='{norm_task}', failed {fail_count}x). Forcing retry.\033[0m"
                )
                avoidance_note = HumanMessage(
                    content=(
                        f"IMPORTANT: The task '{task}' (and all quantity/formatting variants) "
                        f"has been attempted and failed {fail_count} times — it is currently "
                        f"unachievable. Do NOT propose it again under any name variation. "
                        f"Task categories to avoid (failed 3+ times): {over_failed}. "
                        f"Propose an easier or different task that is actually possible right now."
                    )
                )
                return self.propose_next_ai_task(
                    messages=messages + [avoidance_note],
                    max_retries=max_retries - 1,
                )

            context = self.get_task_context(task)
            return task, context
        except Exception as e:
            print(
                f"\033[35mError parsing curriculum response: {e}. Trying again!\033[0m"
            )
            return self.propose_next_ai_task(
                messages=messages,
                max_retries=max_retries - 1,
            )

    def parse_ai_message(self, message):
        task = ""
        for line in message.split("\n"):
            if line.startswith("Task:"):
                task = line[5:].replace(".", "").strip()
        assert task, "Task not found in Curriculum Agent response"
        return {"next_task": task}

    def propose_next_manual_task(self):
        confirmed = False
        task, context = "", ""
        while not confirmed:
            task = input("Enter task: ")
            context = input("Enter context: ")
            print(f"Task: {task}\nContext: {context}")
            confirmed = input("Confirm? (y/n)").lower() in ["y", ""]
        return task, context

    def update_exploration_progress(self, info):
        task = info["task"]
        if task.startswith("Deposit useless items into the chest at"):
            if not info["success"]:
                # extract position string and track failure count
                position = task[len("Deposit useless items into the chest at "):]
                self.failed_deposit_chests[position] = (
                    self.failed_deposit_chests.get(position, 0) + 1
                )
            return
        if info["success"]:
            print(f"\033[35mCompleted task {task}.\033[0m")
            self.completed_tasks.append(task)
        else:
            print(
                f"\033[35mFailed to complete task {task}. Skipping to next task.\033[0m"
            )
            self.failed_tasks.append(task)

        # clean up tasks and dump to disk
        self.clean_up_tasks()

    def clean_up_tasks(self):
        # Keep duplicates in completed_tasks to accurately track repeat counts.
        # Only remove completed tasks from failed_tasks.
        completed_set = set(self.completed_tasks)
        updated_failed_tasks = [t for t in self.failed_tasks if t not in completed_set]

        self.failed_tasks = updated_failed_tasks

        # dump to json
        U.dump_json(
            self.completed_tasks, f"{self.ckpt_dir}/curriculum/completed_tasks.json"
        )
        U.dump_json(self.failed_tasks, f"{self.ckpt_dir}/curriculum/failed_tasks.json")

    def decompose_task(self, task, events):
        messages = [
            SystemMessage(
                content=load_prompt("curriculum_task_decomposition"),
            ),
            self.render_human_message(events=events, chest_observation=""),
            HumanMessage(content=f"Final task: {task}"),
        ]
        print(
            f"\033[31m****Curriculum Agent task decomposition****\nFinal task: {task}\033[0m"
        )
        response = self.llm(messages).content
        print(f"\033[31m****Curriculum Agent task decomposition****\n{response}\033[0m")
        return fix_and_parse_json(response)

    def run_qa(self, *, events, chest_observation):
        questions_new, _ = self.run_qa_step1_ask_questions(
            events=events, chest_observation=chest_observation
        )
        questions = []
        answers = []
        for question in questions_new:
            if self.qa_cache_questions_vectordb._collection.count() > 0:
                docs_and_scores = (
                    self.qa_cache_questions_vectordb.similarity_search_with_score(
                        question, k=1
                    )
                )
                if docs_and_scores and docs_and_scores[0][1] < 0.05:
                    question_cached = docs_and_scores[0][0].page_content
                    assert question_cached in self.qa_cache
                    answer_cached = self.qa_cache[question_cached]
                    questions.append(question_cached)
                    answers.append(answer_cached)
                    continue
            answer = self.run_qa_step2_answer_questions(question=question)
            assert question not in self.qa_cache
            self.qa_cache[question] = answer
            self.qa_cache_questions_vectordb.add_texts(
                texts=[question],
            )
            U.dump_json(self.qa_cache, f"{self.ckpt_dir}/curriculum/qa_cache.json")
            questions.append(question)
            answers.append(answer)
        assert len(questions_new) == len(questions) == len(answers)
        return questions, answers

    def get_task_context(self, task):
        # if include ore in question, gpt will try to use tool with skill touch enhancement to mine
        question = (
            f"How to {task.replace('_', ' ').replace(' ore', '').replace(' ores', '').replace('.', '').strip().lower()}"
            f" in Minecraft?"
        )
        if question in self.qa_cache:
            answer = self.qa_cache[question]
        else:
            answer = self.run_qa_step2_answer_questions(question=question)
            self.qa_cache[question] = answer
            self.qa_cache_questions_vectordb.add_texts(
                texts=[question],
            )
            U.dump_json(self.qa_cache, f"{self.ckpt_dir}/curriculum/qa_cache.json")
        context = f"Question: {question}\n{answer}"
        return context

    def render_system_message_qa_step1_ask_questions(self):
        return SystemMessage(content=load_prompt("curriculum_qa_step1_ask_questions"))

    def render_human_message_qa_step1_ask_questions(self, *, events, chest_observation):
        observation = self.render_observation(
            events=events, chest_observation=chest_observation
        )
        content = ""
        for key in self.curriculum_observations:
            content += observation[key]
        return HumanMessage(content=content)

    def run_qa_step1_ask_questions(self, *, events, chest_observation):
        biome = events[-1][1]["status"]["biome"].replace("_", " ")
        questions = [
            f"What are the blocks that I can find in the {biome} in Minecraft?",
            f"What are the items that I can find in the {biome} in Minecraft?",
            f"What are the mobs that I can find in the {biome} in Minecraft?",
        ]
        concepts = [biome, biome, biome]
        messages = [
            self.render_system_message_qa_step1_ask_questions(),
            self.render_human_message_qa_step1_ask_questions(
                events=events, chest_observation=chest_observation
            ),
        ]
        qa_response = self.qa_llm(messages).content
        try:
            # Regex pattern to extract question and concept pairs
            pattern = r"Question \d+: (.+)\nConcept \d+: (.+)"
            # Extracting all question and concept pairs from the text
            pairs = re.findall(pattern, qa_response)
            # Storing each question and concept in separate lists
            questions_new = [pair[0] for pair in pairs]
            concepts_new = [pair[1] for pair in pairs]
            assert len(questions_new) == len(concepts_new)
            questions.extend(questions_new)
            concepts.extend(concepts_new)
        except Exception as e:
            print(
                f"\033[35mError parsing curriculum response for "
                f"QA step 1 ask questions: {e}.\033[0m"
            )
        return questions, concepts

    def render_system_message_qa_step2_answer_questions(self):
        return SystemMessage(
            content=load_prompt("curriculum_qa_step2_answer_questions")
        )

    def render_human_message_qa_step2_answer_questions(self, question):
        content = f"Question: {question}"
        return HumanMessage(content=content)

    def run_qa_step2_answer_questions(self, question):
        messages = [
            self.render_system_message_qa_step2_answer_questions(),
            self.render_human_message_qa_step2_answer_questions(question=question),
        ]
        print(f"\033[35mCurriculum Agent Question: {question}\033[0m")
        qa_answer = self.qa_llm(messages).content
        print(f"\033[31mCurriculum Agent {qa_answer}\033[0m")
        return qa_answer
