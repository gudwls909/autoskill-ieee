import os

import voyager.utils as U
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_chroma import Chroma

from voyager.prompts import load_prompt
from voyager.control_primitives import load_control_primitives


class SkillManager:
    def __init__(
        self,
        model_name="gpt-3.5-turbo",
        temperature=0,
        retrieval_top_k=5,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        log_dir=None,
    ):
        if log_dir is not None:
            token_logger = U.TokenLogger(
                agent_name="skill",
                log_path=os.path.join(log_dir, "token_log.jsonl"),
            )
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                request_timeout=request_timout,
                callbacks=[token_logger],
            )
        else:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                request_timeout=request_timout,
            )
        U.f_mkdir(f"{ckpt_dir}/skill/code")
        U.f_mkdir(f"{ckpt_dir}/skill/description")
        U.f_mkdir(f"{ckpt_dir}/skill/vectordb")
        U.f_mkdir(f"{ckpt_dir}/skill/semantic_vectordb")
        # programs for env execution
        self.control_primitives = load_control_primitives()
        if resume:
            print(f"\033[33mLoading Skill Manager from {ckpt_dir}/skill\033[0m")
            self.skills = U.load_json(f"{ckpt_dir}/skill/skills.json")
            _fi_path = f"{ckpt_dir}/skill/family_index.json"
            self.family_index = U.load_json(_fi_path) if os.path.exists(_fi_path) else {}
        else:
            self.skills = {}
            self.family_index = {}
        self.retrieval_top_k = retrieval_top_k
        self.ckpt_dir = ckpt_dir
        self.last_retrieved_names = []

        if log_dir is not None:
            embedding_fn = U.LoggedEmbeddings(
                agent_name="skill_embed",
                log_path=os.path.join(log_dir, "embedding_log.jsonl"),
            )
            sem_embedding_fn = U.LoggedEmbeddings(
                agent_name="skill_sem_embed",
                log_path=os.path.join(log_dir, "embedding_log.jsonl"),
            )
        else:
            embedding_fn = OpenAIEmbeddings()
            sem_embedding_fn = OpenAIEmbeddings()

        self.vectordb = Chroma(
            collection_name="skill_vectordb",
            embedding_function=embedding_fn,
            persist_directory=f"{ckpt_dir}/skill/vectordb",
        )
        self.semantic_vectordb = Chroma(
            collection_name="skill_semantic_vectordb",
            embedding_function=sem_embedding_fn,
            persist_directory=f"{ckpt_dir}/skill/semantic_vectordb",
        )

        self._repair_vectordb(self.vectordb, "description")
        self._repair_semantic_vectordb()

    # ------------------------------------------------------------------
    # Sync / repair
    # ------------------------------------------------------------------

    def _repair_vectordb(self, vectordb, text_field):
        """Ensure vectordb count matches skills. Adds missing entries if needed."""
        vdb_count = vectordb._collection.count()
        if vdb_count == len(self.skills):
            return
        print(
            f"\033[33mSkill Manager: vectordb out of sync "
            f"({vdb_count} vs {len(self.skills)}). Repairing...\033[0m"
        )
        existing_ids = set(vectordb._collection.get()["ids"]) if vdb_count > 0 else set()
        for name, entry in self.skills.items():
            if name not in existing_ids:
                text = entry.get(text_field) or entry.get("description", "")
                if text:
                    vectordb.add_texts(
                        texts=[text],
                        ids=[name],
                        metadatas=[{"name": name}],
                    )
        assert vectordb._collection.count() == len(self.skills), (
            f"Skill Manager's vectordb could not be repaired.\n"
            f"There are {vectordb._collection.count()} skills in vectordb "
            f"but {len(self.skills)} skills in skills.json.\n"
            f"You may need to manually delete the vectordb directory."
        )

    def _repair_semantic_vectordb(self):
        """Rebuild semantic_vectordb entries from skills that have canonical_desc."""
        vdb_count = self.semantic_vectordb._collection.count()
        skills_with_desc = {
            name for name, e in self.skills.items() if e.get("canonical_desc")
        }
        if vdb_count == len(skills_with_desc):
            return
        print(
            f"\033[33mSkill Manager: semantic_vectordb out of sync "
            f"({vdb_count} vs {len(skills_with_desc)}). Repairing...\033[0m"
        )
        existing_ids = (
            set(self.semantic_vectordb._collection.get()["ids"])
            if vdb_count > 0
            else set()
        )
        for name in skills_with_desc:
            if name not in existing_ids:
                desc = self.skills[name]["canonical_desc"]
                self.semantic_vectordb.add_texts(
                    texts=[desc],
                    ids=[name],
                    metadatas=[{"name": name}],
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def programs(self):
        programs = ""
        for skill_name, entry in self.skills.items():
            programs += f"{entry['code']}\n\n"
        for primitives in self.control_primitives:
            programs += f"{primitives}\n\n"
        return programs

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_new_skill(self, info, canonical=None, allow_dedup: bool = True):
        """Add a skill to the library.

        Returns a dict:
          {"status": "added",   "representative": program_name}
          {"status": "dedup",   "representative": existing_name}   # family already present
          {"status": "skipped", "representative": None}             # deposit task etc.

        allow_dedup=False disables the deduplication gate (ablation study).
        """
        if info["task"].startswith("Deposit useless items into the chest at"):
            return {"status": "skipped", "representative": None}

        program_name = info["program_name"]
        program_code = info["program_code"]

        # --- Contribution 2: dedup gate (skipped when allow_dedup=False) ---
        if allow_dedup and canonical:
            family_key = canonical.get("family_key", "")
            representative = self.family_index.get(family_key) if family_key else None
            if representative:
                if representative != program_name:
                    # Different function name claims same family → dedup
                    print(
                        f"\033[33mSkill Manager: dedup — {program_name} "
                        f"is a variant of existing skill {representative} "
                        f"(family: {family_key}). Skipping.\033[0m"
                    )
                    return {"status": "dedup", "representative": representative}
                else:
                    # Same function name IS the representative and already in library → always dedup.
                    # Covers exact repeats (count=4→4) and count variants (count=4→6) alike.
                    if program_name in self.skills:
                        existing_count = (
                            (self.skills[program_name].get("canonical_json") or {})
                            .get("requested_count")
                        )
                        new_count = (canonical.get("canonical_json") or {}).get("requested_count")
                        print(
                            f"\033[33mSkill Manager: dedup — {program_name} already registered "
                            f"as representative for family {family_key} "
                            f"(existing count={existing_count}, new count={new_count}). Skipping.\033[0m"
                        )
                        return {"status": "dedup", "representative": representative}

        skill_description = self.generate_skill_description(program_name, program_code)
        print(
            f"\033[33mSkill Manager generated description for {program_name}:\n{skill_description}\033[0m"
        )
        if program_name in self.skills:
            print(f"\033[33mSkill {program_name} already exists. Rewriting!\033[0m")
            self.vectordb._collection.delete(ids=[program_name])
            i = 2
            while f"{program_name}V{i}.js" in os.listdir(f"{self.ckpt_dir}/skill/code"):
                i += 1
            dumped_program_name = f"{program_name}V{i}"
        else:
            dumped_program_name = program_name

        # --- prompt vectordb ---
        self.vectordb.add_texts(
            texts=[skill_description],
            ids=[program_name],
            metadatas=[{"name": program_name}],
        )
        entry = {
            "code": program_code,
            "description": skill_description,
        }

        # --- canonical / semantic bookkeeping ---
        if canonical:
            family_key = canonical.get("family_key", "")
            canonical_desc = canonical.get("canonical_desc", "")

            entry["canonical_json"] = canonical.get("canonical_json") or {}
            entry["canonical_desc"] = canonical_desc
            entry["family_key"] = family_key
            entry["parse_mode"] = canonical.get("parse_mode", "")

            # semantic vectordb (index by canonical_desc for similarity search)
            if canonical_desc and program_name not in (
                set(self.semantic_vectordb._collection.get()["ids"])
                if self.semantic_vectordb._collection.count() > 0
                else set()
            ):
                self.semantic_vectordb.add_texts(
                    texts=[canonical_desc],
                    ids=[program_name],
                    metadatas=[{"name": program_name}],
                )

            # family_index: first skill to claim a family_key wins
            if family_key and family_key not in self.family_index:
                self.family_index[family_key] = program_name
                U.dump_json(self.family_index, f"{self.ckpt_dir}/skill/family_index.json")

        self.skills[program_name] = entry
        assert self.vectordb._collection.count() == len(self.skills), (
            "vectordb is not synced with skills.json"
        )
        U.dump_text(program_code, f"{self.ckpt_dir}/skill/code/{dumped_program_name}.js")
        U.dump_text(
            skill_description,
            f"{self.ckpt_dir}/skill/description/{dumped_program_name}.txt",
        )
        U.dump_json(self.skills, f"{self.ckpt_dir}/skill/skills.json")
        return {"status": "added", "representative": program_name}

    def generate_skill_description(self, program_name, program_code):
        messages = [
            SystemMessage(content=load_prompt("skill")),
            HumanMessage(
                content=program_code
                + "\n\n"
                + f"The main function is `{program_name}`."
            ),
        ]
        skill_description = f"    // { self.llm(messages).content}"
        return f"async function {program_name}(bot) {{\n{skill_description}\n}}"

    def retrieve_skills(self, query):
        total = self.vectordb._collection.count()
        k = min(total, self.retrieval_top_k)
        if k == 0:
            self.last_retrieved_names = []
            return []
        print(f"\033[33mSkill Manager retrieving {k}/{total} skills\033[0m")
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        self.last_retrieved_names = [doc.metadata["name"] for doc, _ in docs_and_scores]
        print(
            f"\033[33mSkill Manager retrieved skills: "
            f"{', '.join(self.last_retrieved_names)}\033[0m"
        )
        skills = []
        for doc, _ in docs_and_scores:
            skills.append(self.skills[doc.metadata["name"]]["code"])
        return skills

    # ------------------------------------------------------------------
    # Semantic queries (Contribution 2)
    # ------------------------------------------------------------------

    def semantic_neighbors(self, text: str, k: int = 3):
        """Return [(skill_name, score)] for the k most semantically similar skills."""
        count = self.semantic_vectordb._collection.count()
        if count == 0:
            return []
        k = min(k, count)
        docs_and_scores = self.semantic_vectordb.similarity_search_with_score(text, k=k)
        return [(doc.metadata["name"], score) for doc, score in docs_and_scores]

    def lookup_family(self, family_key: str):
        """Return the representative skill name for a family_key, or None."""
        return self.family_index.get(family_key)
