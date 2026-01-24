import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from metagpt.actions import Action
from metagpt.logs import logger
from utils.global_vars import PROMPT_BASE_DIR
from utils.global_vars import DATASET_NAME
from collections import OrderedDict


class A5_ReasoningJudge_Light(Action):
    """
    Lightweight A5 for ProntoQA and other 2-value logic datasets.
    - Uses QuickReflection for uncertainty resolution
    - On conflict (both True/True or False/False), fallback to S1's verdict
    - Skips DeepReflection to optimize for speed
    """
    name: str = "A5_ReasoningJudge_Light"

    async def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        semiotic_square: dict = None,
        context: str = "",
        premises: Any = None
    ) -> dict:
        if inputs:
            semiotic_square = inputs.get("semiotic_square", semiotic_square)
            context = inputs.get("context", context)
            premises = inputs.get("premises", premises)

        self.global_premises = premises

        def normalize_square(square):
            key_aliases = {"¬S1": "not_S1", "¬S2": "not_S2"}
            square = dict(square)
            for unicode_key, ascii_key in key_aliases.items():
                if unicode_key in square and ascii_key not in square:
                    square[ascii_key] = square[unicode_key]
            for k in {"S1", "S2", "not_S1", "not_S2"}:
                if k in square and not isinstance(square[k], dict):
                    square[k] = {"statement": square[k], "FOL": square[k]}
            return square

        semiotic_square = normalize_square(semiotic_square)
        s2_type = semiotic_square.get("S2_type", "").lower()
        trace = {}

        # Step 1: Run S1 and not_S1 reasoning pipeline
        trace["S1"] = await self._run_reasoning_pipeline("S1", context, semiotic_square)
        s1_verdict = trace["S1"].get("verdict", "Uncertain")

        trace["not_S1"] = await self._run_reasoning_pipeline("not_S1", context, semiotic_square)
        not_s1_verdict = trace["not_S1"].get("verdict", "Uncertain")

        return await self._verify_and_return_light(semiotic_square, trace, s2_type, s1_verdict, not_s1_verdict)

    async def _run_reasoning_pipeline(self, statement_key: str, context: str, square: dict) -> dict:
        """Reuses the same pipeline logic as the full version but with global premises optimization."""
        statement = square.get(statement_key, {})

        # Construct target_statement with both statement and FOL
        stmt_text = statement.get("statement", "")
        fol = statement.get("FOL", "")
        target_statement = f"Statement: {stmt_text}\nFOL: {fol}"

        # A4: Reuse global premises
        if hasattr(self, 'global_premises') and self.global_premises is not None:
            logger.info(f"[A5_Light] ✅ Reusing global premises for {statement_key}")
            premises = self.global_premises
        else:
            logger.info(f"[A5_Light] ⚠️  Extracting premises via A4 for {statement_key}")
            a4_path = os.path.join(PROMPT_BASE_DIR, "A4_ContextHandle.txt")
            try:
                with open(a4_path, "r", encoding="utf-8") as f:
                    a4_prompt = f.read()
            except Exception as e:
                logger.error(f"[A4] Failed to load prompt from {a4_path}: {e}")
                return {"error": str(e), "verdict": "Uncertain"}

            a4_filled = a4_prompt.replace("{context}", context.strip()).replace("{target_statement}", target_statement)
            a4_rsp = await self._log_and_ask(a4_filled)

            try:
                match = re.search(r"```json(.*?)```", a4_rsp, re.DOTALL)
                premises = json.loads(match.group(1).strip())
            except Exception as e:
                logger.warning(f"[A5_Light] A4_ContextHandle parsing failed: {e}")
                return {"verdict": "Uncertain", "steps": [], "error": f"A4_ContextHandle error: {e}"}

        # A5_Plan
        plan_path = os.path.join(PROMPT_BASE_DIR, "A5_Plan.txt")
        with open(plan_path, "r", encoding="utf-8") as f:
            plan_prompt = f.read()
        plan_filled = plan_prompt.replace("{target_statement}", target_statement) \
                                .replace("{premises}", self._format_premises(dataset_name=DATASET_NAME, premises=premises))
        plan_rsp = await self._log_and_ask(plan_filled)

        try:
            match = re.search(r"```json(.*?)```", plan_rsp, re.DOTALL)
            plan_json = json.loads(match.group(1).strip())
            plan_steps = plan_json.get("plan", [])
        except Exception as e:
            logger.warning(f"[A5_Light] Plan parsing failed: {e}")
            return {"verdict": "Uncertain", "steps": [], "error": f"Plan error: {e}"}

        # A5_Reasoning
        reasoning_path = os.path.join(PROMPT_BASE_DIR, "A5_Reasoning.txt")
        with open(reasoning_path, "r", encoding="utf-8") as f:
            reasoning_prompt = f.read()

        formatted_premises = self._format_premises(dataset_name=DATASET_NAME, premises=premises)
        filled_reasoning = reasoning_prompt.replace("{premises}", formatted_premises) \
                                        .replace("{target_statement}", target_statement) \
                                        .replace("{PLAN}", json.dumps(plan_steps, indent=2))
        rsp = await self._log_and_ask(filled_reasoning)

        try:
            match = re.search(r"```json(.*?)```", rsp, re.DOTALL)
            reasoning_json = json.loads(match.group(1).strip())
            verdict = reasoning_json.get("verdict", "Uncertain")
            steps = reasoning_json.get("steps", [])
        except Exception as e:
            logger.warning(f"[A5_Light] Reasoning parsing failed: {e}")
            return {"verdict": "Uncertain", "steps": [], "error": f"Reasoning error: {e}"}

        result = OrderedDict()
        result["statement"] = stmt_text
        result["FOL"] = fol
        result["premises"] = premises
        result["plan_steps"] = plan_steps
        result["reasoning_steps"] = steps
        result["verdict"] = verdict

        return result

    async def _verify_and_return_light(self, square, trace, s2_type, s1_verdict, not_s1_verdict):
        """Lightweight verification: QuickReflection only, fallback to S1 on conflicts."""
        # 1. Direct Resolution
        direct_result = self.direct_resolution(s1_verdict, not_s1_verdict)
        if direct_result is not None:
            trace["final_decision"] = {"method": "Direct Resolution", "verdict": direct_result}
            return {"s1_truth": direct_result, "reasoning_trace": trace}

        # 2. Quick Reflection (for uncertainties)
        if s1_verdict == "Uncertain" or not_s1_verdict == "Uncertain":
            quick_result = await self.quick_reflection(trace, square, s1_verdict, not_s1_verdict)
            if quick_result is not None:
                trace["final_decision"] = {"method": "Quick Reflection", "verdict": quick_result}
                return {"s1_truth": quick_result, "reasoning_trace": trace}

        # 3. Conflict Fallback (ProntoQA-specific: trust S1 on conflicts)
        if s1_verdict == not_s1_verdict and s1_verdict != "Uncertain":
            logger.info(f"[A5_Light] Conflict detected (both {s1_verdict}), falling back to S1's verdict")
            trace["final_decision"] = {"method": "Conflict Fallback to S1", "verdict": s1_verdict}
            return {"s1_truth": s1_verdict, "reasoning_trace": trace}

        # Fallback
        trace["final_decision"] = {"method": "Fallback", "verdict": "Uncertain"}
        return {"s1_truth": "Uncertain", "reasoning_trace": trace}

    def direct_resolution(self, s1, not_s1):
        if s1 == "True" and not_s1 == "False": return "True"
        if s1 == "False" and not_s1 == "True": return "False"
        if s1 == "Uncertain" and not_s1 == "Uncertain": return "Uncertain"
        return None

    async def quick_reflection(self, trace, square, s1_verdict, not_s1_verdict):
        s1_block = trace.get("S1", {})
        not_s1_block = trace.get("not_S1", {})
        s2_type = square.get("S2_type", "")

        reflection_verdict, reflection_reasoning = await self.run_reflection_pass(
            s1_block=s1_block,
            s2_block=not_s1_block,
            s2_type=s2_type,
            initial_verdict=s1_verdict
        )

        trace["quick_reflection_detail"] = reflection_reasoning
        return reflection_verdict

    async def run_reflection_pass(self, s1_block: dict, s2_block: dict, s2_type: str, initial_verdict: str) -> Tuple[str, dict]:
        verify_path = os.path.join(PROMPT_BASE_DIR, "A5_Verify.txt")
        try:
            with open(verify_path, "r", encoding="utf-8") as f:
                verify_prompt = f.read()
        except Exception as e:
            logger.warning(f"[A5_Light] Failed to load A5_Verify.txt: {e}")
            return "Uncertain", {"error": str(e)}

        execution_block = f"### S1\n```json\n{json.dumps(s1_block, indent=2)}\n```\n\nS2_type: \"{s2_type}\"\n\n### not_S1\n```json\n{json.dumps(s2_block, indent=2)}\n```"
        prompt_filled = verify_prompt.replace("[[EXECUTION]]", execution_block)
        rsp = await self._log_and_ask(prompt_filled)

        try:
            match = re.search(r"```json(.*?)```", rsp, re.DOTALL)
            reflection = json.loads((match.group(1) if match else rsp).strip())
            return reflection.get("verdict", "Uncertain"), reflection
        except Exception as e:
            logger.warning(f"[A5_Light] Reflection parsing failed: {e}")
            return "Uncertain", {"error": str(e), "raw_response": rsp}

    def _format_premises(self, dataset_name: str, premises: Any) -> str:
        if not premises: return "No explicit premises provided."
        formatted = []
        def format_line(i, item):
            if isinstance(item, dict):
                stmt = item.get('statement', '').strip()
                fol = item.get('FOL', '').strip()
                return f"{i + 1}. {stmt}\nFOL: {fol}"
            else:
                parts = item.split(":::")
                fol = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                return f"{i + 1}. {fol}\nDescription: {desc}"

        if dataset_name in {"FOLIO", "ProverQA"}:
            for section in ["Predicates", "Premises", "Conclusion"]:
                items = premises.get(section, [])
                if items:
                    formatted.append(f"### {section}:")
                    for i, item in enumerate(items): formatted.append(format_line(i, item))
                    formatted.append("")
        elif dataset_name == "ProofWriter":
            for section in ["Predicates", "Facts", "Rules", "Conditional rules", "Rules with compound predicates by comma"]:
                items = premises.get(section, [])
                if items:
                    formatted.append(f"### {section}:")
                    for i, item in enumerate(items): formatted.append(format_line(i, item))
                    formatted.append("")
        elif dataset_name in {"RepublicQA", "ProntoQA"}:
            if isinstance(premises, list):
                formatted.append("### Premises:")
                for i, item in enumerate(premises): formatted.append(format_line(i, item))
            else:
                for section, items in premises.items():
                    if items:
                        formatted.append(f"### {section}:")
                        for i, item in enumerate(items): formatted.append(format_line(i, item))
                        formatted.append("")
        return "\n".join(formatted)

    async def _log_and_ask(self, prompt: str) -> str:
        logger.debug(f"==== Prompt to LLM ====\n{prompt}")
        rsp = await self._aask(prompt)
        logger.debug(f"==== Raw LLM Response ====\n{rsp}")
        return rsp
