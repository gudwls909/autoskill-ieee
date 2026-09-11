import os
import re
from typing import Any, Dict, List, Optional

import voyager.utils as U
from voyager.prompts import load_prompt

try:
    from langchain_openai import ChatOpenAI
except Exception:
    from langchain.chat_models import ChatOpenAI

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:
    from langchain.schema import HumanMessage, SystemMessage

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}

_PRIMITIVES = {"mineBlock", "craftItem", "smeltItem", "placeItem", "killMob", "exploreUntil"}
_PRIMITIVE_TO_VERB = {
    "mineBlock": "mine", "craftItem": "craft", "smeltItem": "smelt",
    "placeItem": "place", "killMob": "kill",
}
# Normalize task-text verb → canonical primitive name for family_key
_VERB_TO_PRIMITIVE = {
    "mine": "mineBlock", "craft": "craftItem", "smelt": "smeltItem",
    "place": "placeItem", "kill": "killMob",
}
# Explicit target aliases: item-drop names / plurals → canonical block/item name
_TARGET_ALIASES = {
    "iron":            "iron_ore",
    "raw_iron":        "iron_ore",
    "copper":          "copper_ore",
    "raw_copper":      "copper_ore",
    "gold":            "gold_ore",
    "raw_gold":        "gold_ore",
    "coal":            "coal_ore",
    "diamond":         "diamond_ore",
    "emerald":         "emerald_ore",
    "redstone":        "redstone_ore",
    "lapis":           "lapis_ore",
    "lapis_lazuli":    "lapis_ore",
    "quartz":          "nether_quartz_ore",
    "stone_block":     "cobblestone",
    "stone_blocks":    "cobblestone",
    "wood_log":        "oak_log",
    "wood_logs":       "oak_log",
}


class SkillSemanticAgent:
    """
    Canonicalizes a skill into a structured JSON representation.

    Two-pass strategy:
      1. Rule-based fast path  (no LLM cost) for simple single-primitive skills
      2. LLM fallback          for complex / multi-step skills
    """

    def __init__(
        self,
        log_dir: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        request_timout: int = 120,
        enable_llm: Optional[bool] = None,
    ):
        self.log_dir = log_dir
        if enable_llm is None:
            enable_llm = os.getenv("SKILL_SEMANTIC_LLM_ENABLE", "1") == "1"
        self.enable_llm = enable_llm

        token_logger = U.TokenLogger(
            agent_name="skill_semantic",
            log_path=os.path.join(log_dir, "token_log.jsonl"),
        )
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
            callbacks=[token_logger],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def canonicalize(
        self,
        program_name: str,
        program_code: str,
        task_text: Optional[str] = None,
        allow_llm: bool = True,
    ) -> Dict[str, Any]:
        result = self._rule(program_name, program_code, task_text)
        if result is not None:
            return result

        if allow_llm and self.enable_llm:
            result = self._llm(program_name, program_code, task_text)
            if result is not None:
                return result

        return self._fallback(program_name, program_code, task_text)

    # ------------------------------------------------------------------
    # Rule-based path
    # ------------------------------------------------------------------

    def _rule(self, program_name, program_code, task_text) -> Optional[Dict]:
        pcalls = self._extract_primitive_calls(program_code)
        scalls = self._extract_skill_calls(program_code, program_name)
        task_spec = self._parse_task_text(task_text)

        if not pcalls:
            return None

        # exploreUntil is always auxiliary — strip it for the purpose of classifying the skill
        actual_pcalls = [c for c in pcalls if c["primitive"] != "exploreUntil"]
        terminal = actual_pcalls[-1] if actual_pcalls else (pcalls[-1] if pcalls else None)
        if terminal is None:
            return None
        terminal_prim = terminal["primitive"]

        # Allow single actual-primitive (with or without leading exploreUntil),
        # OR terminal-workflow (craft/smelt/mine/kill at end preceded by setup calls)
        is_terminal_workflow = (
            len(scalls) == 0
            and len(actual_pcalls) >= 1
        )
        if not actual_pcalls or (len(actual_pcalls) != 1 and not is_terminal_workflow):
            return None

        count = task_spec.get("count") or terminal["args"].get("count")
        target = task_spec.get("target") or terminal["args"].get("target")
        verb = task_spec.get("verb") or _PRIMITIVE_TO_VERB.get(terminal_prim, terminal_prim)
        is_atomic = (len(actual_pcalls) == 1 and len(scalls) == 0)

        cjson = self._build_canonical_json(
            program_name=program_name,
            summary_type="single_primitive" if len(pcalls) == 1 else "terminal_primitive_workflow",
            pcalls=pcalls,
            scalls=scalls,
            count=count,
            target=target,
            verb=verb,
            is_atomic=is_atomic,
        )
        return {
            "canonical_json": cjson,
            "canonical_desc": self._desc(cjson),
            "family_key": self._family_key(cjson),
            "complexity": self._complexity(program_code, pcalls),
            "parse_mode": "rule" if len(actual_pcalls) == 1 else "rule_terminal_workflow",
            "skill_calls": scalls,
        }

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _llm(self, program_name, program_code, task_text) -> Optional[Dict]:
        messages = [
            SystemMessage(content=load_prompt("skill_semantic")),
            HumanMessage(content=(
                f"Task text: {task_text}\n"
                f"Program name: {program_name}\n"
                f"Program code:\n```javascript\n{program_code}\n```"
            )),
        ]
        try:
            response = self.llm.invoke(messages)
            parsed = U.fix_and_parse_json(response.content)
            if not isinstance(parsed, dict):
                return None
        except Exception:
            return None

        cjson = parsed.get("canonical_json")
        canonical_desc = parsed.get("canonical_desc", "")
        complexity = parsed.get("complexity")

        if not isinstance(cjson, dict) or not canonical_desc:
            return None

        # Fill in any missing fields
        pcalls = self._extract_primitive_calls(program_code)
        scalls = self._extract_skill_calls(program_code, program_name)
        task_spec = self._parse_task_text(task_text)

        cjson.setdefault("skill_calls", scalls)
        cjson.setdefault("program_name", program_name)
        cjson.setdefault("is_atomic_single_primitive", len(pcalls) == 1 and len(scalls) == 0)

        requires = cjson.get("requires")
        if not isinstance(requires, dict):
            requires = _empty_requires()
        requires["skills"] = requires.get("skills") or scalls
        cjson["requires"] = requires

        produces = cjson.get("produces")
        if not isinstance(produces, dict):
            produces = self._infer_produces(pcalls)
        cjson["produces"] = produces

        for k, fallback in [
            ("requested_count", task_spec.get("count")),
            ("requested_target", task_spec.get("target")),
            ("requested_verb", task_spec.get("verb")),
        ]:
            if not cjson.get(k):
                cjson[k] = fallback
        cjson["variant_kind"] = "count_variant" if cjson.get("requested_count") else "generic"

        # Always derive family_key from structured cjson — LLM often returns function names
        family_key = self._family_key(cjson)
        if not isinstance(complexity, dict):
            complexity = self._complexity(program_code, pcalls)

        return {
            "canonical_json": cjson,
            "canonical_desc": canonical_desc.strip(),
            "family_key": family_key,
            "complexity": complexity,
            "parse_mode": "llm",
            "skill_calls": scalls,
        }

    # ------------------------------------------------------------------
    # Fallback (no LLM, rule failed)
    # ------------------------------------------------------------------

    def _fallback(self, program_name, program_code, task_text) -> Dict:
        pcalls = self._extract_primitive_calls(program_code)
        scalls = self._extract_skill_calls(program_code, program_name)
        task_spec = self._parse_task_text(task_text)

        cjson = self._build_canonical_json(
            program_name=program_name,
            summary_type="multi_step",
            pcalls=pcalls,
            scalls=scalls,
            count=task_spec.get("count"),
            target=task_spec.get("target"),
            verb=task_spec.get("verb"),
            is_atomic=(len(pcalls) == 1 and len(scalls) == 0),
        )
        return {
            "canonical_json": cjson,
            "canonical_desc": self._desc(cjson),
            "family_key": self._family_key(cjson),
            "complexity": self._complexity(program_code, pcalls),
            "parse_mode": "fallback",
            "skill_calls": scalls,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_canonical_json(
        self, program_name, summary_type, pcalls, scalls, count, target, verb, is_atomic
    ) -> Dict:
        return {
            "program_name": program_name,
            "summary_type": summary_type,
            "primitive_calls": [
                {
                    "name": c["primitive"],
                    "target": c["args"].get("target"),
                    "target_type": c["args"].get("target_type"),
                    "count": c["args"].get("count"),
                    "timeout": c["args"].get("timeout"),
                }
                for c in pcalls
            ],
            "skill_calls": scalls,
            "requires": {
                "skills": scalls,
                "items": [], "tools": [], "blocks": [], "stations": [],
            },
            "produces": self._infer_produces(pcalls),
            "requested_count": count,
            "requested_target": target,
            "requested_verb": verb,
            "variant_kind": "count_variant" if count is not None else "generic",
            "is_atomic_single_primitive": is_atomic,
        }

    def _desc(self, cjson: Dict) -> str:
        calls = cjson.get("primitive_calls") or []
        scalls = cjson.get("skill_calls") or []
        target = cjson.get("requested_target") or (calls[-1].get("target") if calls else None) or "target"
        verb = cjson.get("requested_verb") or (
            _PRIMITIVE_TO_VERB.get(calls[-1].get("name", ""), "") if calls else ""
        )

        composite = bool(scalls) or len(calls) > 1

        if verb == "mine" or (calls and calls[-1].get("name") == "mineBlock"):
            return f"Obtain {target} by mining, using prerequisite setup if needed." if composite else f"Mine {target}."
        if verb == "craft" or (calls and calls[-1].get("name") == "craftItem"):
            return f"Craft {target}, gathering required materials and setup if needed." if composite else f"Craft {target}."
        if verb == "smelt" or (calls and calls[-1].get("name") == "smeltItem"):
            return f"Smelt {target}, ensuring furnace and fuel if needed." if composite else f"Smelt {target}."
        if verb == "place" or (calls and calls[-1].get("name") == "placeItem"):
            return f"Place {target}."
        if verb == "kill" or (calls and calls[-1].get("name") == "killMob"):
            return f"Kill {target}."
        if target != "target":
            return f"Complete a multi-step workflow for {target}."
        return "Execute a reusable multi-step workflow."

    def _family_key(self, cjson: Dict) -> str:
        calls = cjson.get("primitive_calls") or []
        scalls = cjson.get("skill_calls") or []
        actual_calls = [c for c in calls if c.get("name") != "exploreUntil"]

        # Determine canonical primitive verb and target
        if actual_calls:
            c = actual_calls[-1]
            prim = c.get("name", "unknown")
            target = self._normalize_target(c.get("target")) or "*"
        else:
            prim = None
            target = None

        # Prefer target from requested_target (task text), fall back to code
        req_target = self._normalize_target(cjson.get("requested_target"))
        req_verb = cjson.get("requested_verb")
        if req_target:
            target = req_target
        # Normalize verb (mine/craft/smelt) → primitive name (mineBlock/craftItem/smeltItem)
        if req_verb:
            prim = _VERB_TO_PRIMITIVE.get(req_verb, prim)
        if prim and target:
            return f"{prim}:{target}:*"
        if prim:
            return f"{prim}:*:*"
        return "unknown:*"

    def _complexity(self, code: str, pcalls: List) -> Dict:
        return {
            "num_primitive_calls": len(pcalls),
            "has_exploreUntil": any(c["primitive"] == "exploreUntil" for c in pcalls),
            "primitive_names": [c["primitive"] for c in pcalls],
            "code_lines": len([l for l in code.splitlines() if l.strip()]),
        }

    def _infer_produces(self, pcalls: List) -> Dict:
        items, blocks, states = [], [], []
        for c in pcalls:
            p = c.get("primitive")
            t = c.get("args", {}).get("target")
            if not t:
                continue
            if p in {"craftItem", "smeltItem", "placeItem"}:
                if t not in items:
                    items.append(t)
            elif p == "mineBlock":
                if t not in blocks:
                    blocks.append(t)
            elif p == "killMob":
                s = f"{t}_killed"
                if s not in states:
                    states.append(s)
        return {"items": items, "blocks": blocks, "states": states}

    def _extract_primitive_calls(self, code: str) -> List[Dict]:
        pattern = re.compile(
            r"await\s+(mineBlock|craftItem|smeltItem|placeItem|killMob|exploreUntil)\s*\((.*?)\)\s*;",
            re.DOTALL,
        )
        calls = []
        for prim, arg_blob in pattern.findall(code):
            raw_args = [p.strip() for p in arg_blob.split(",")]
            calls.append({"primitive": prim, "raw_args": raw_args, "args": self._parse_args(prim, raw_args)})
        return calls

    def _extract_skill_calls(self, code: str, program_name: Optional[str] = None) -> List[str]:
        pattern = re.compile(r"await\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*bot\b")
        found = []
        for fn in pattern.findall(code):
            if fn in _PRIMITIVES:
                continue
            if program_name and fn == program_name:
                continue
            if fn not in found:
                found.append(fn)
        return found

    def _parse_args(self, primitive: str, raw_args: List[str]) -> Dict:
        args = {"target": None, "target_type": None, "count": None, "timeout": None}
        target = self._normalize_target(self._strip_quotes(raw_args[1]) if len(raw_args) >= 2 else None)

        if primitive == "mineBlock":
            args.update(target_type="block", target=target)
            if len(raw_args) >= 3:
                args["count"] = self._parse_number(raw_args[2])
        elif primitive in {"craftItem", "smeltItem", "placeItem"}:
            args.update(target_type="item", target=target)
            if len(raw_args) >= 3:
                args["count"] = self._parse_number(raw_args[2])
        elif primitive == "killMob":
            args.update(target_type="mob", target=target)
            if len(raw_args) >= 3:
                args["timeout"] = self._parse_number(raw_args[2])
        elif primitive == "exploreUntil":
            args["target_type"] = "predicate"
        return args

    @staticmethod
    def _normalize_target(target: Optional[str]) -> Optional[str]:
        """Canonicalize target names: strip raw_ prefix, singularize ore/log plurals, apply aliases."""
        if not target:
            return target
        t = target.strip().lower().replace(" ", "_").replace("-", "_")
        # Strip "raw_" prefix (raw_iron_ores → iron_ores → iron_ore)
        if t.startswith("raw_"):
            t = t[4:]
        # Singularize common Minecraft plural forms
        for suffix, singular in [("_ores", "_ore"), ("_logs", "_log"), ("_ingots", "_ingot")]:
            if t.endswith(suffix):
                t = t[: -len(suffix)] + singular
                break
        # Explicit alias table
        return _TARGET_ALIASES.get(t, t)

    def _parse_task_text(self, task_text: Optional[str]) -> Dict:
        if not task_text:
            return {}
        # Match "Verb N target" (e.g. "Mine 3 iron ore") OR
        # "Verb a/an target" (e.g. "Craft an iron chestplate") treating a/an as count=1
        m = re.match(
            r"^(Mine|Craft|Smelt|Place|Kill)\s+(\d+|a|an)\s+(.+)$",
            task_text.strip(), re.IGNORECASE
        )
        if not m:
            return {}
        count_raw = m.group(2).lower()
        count = 1 if count_raw in ("a", "an") else int(count_raw)
        raw_target = re.sub(r"[\s\-]+", "_", m.group(3).strip().lower())
        return {
            "verb": m.group(1).lower(),
            "count": count,
            "target": self._normalize_target(raw_target),
        }

    def canonicalize_task(self, task_text: str) -> Optional[Dict]:
        """Rule-based family lookup from task text alone (no code needed).
        Returns {"family_key": ..., "canonical_json": {...}} or None if unparseable."""
        parsed = self._parse_task_text(task_text)
        if not parsed:
            return None
        cjson = {
            "requested_verb": parsed["verb"],
            "requested_target": parsed["target"],
            "requested_count": parsed["count"],
        }
        fk = self._family_key(cjson)
        if not fk or fk.startswith("unknown:"):
            return None
        return {"family_key": fk, "canonical_json": cjson}

    def _parse_number(self, raw: str) -> Optional[int]:
        s = raw.strip().strip(",").strip("\"'")
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        return _NUMBER_WORDS.get(s.lower())

    def _strip_quotes(self, raw: str) -> Optional[str]:
        m = re.fullmatch(r"['\"]([^'\"]+)['\"]", raw.strip())
        return m.group(1) if m else None


def _empty_requires() -> Dict:
    return {"skills": [], "items": [], "tools": [], "blocks": [], "stations": []}
