# PhilAgent/actions/A4_context_handle.py

import os
import re
import json
from typing import ClassVar, Dict, Any, Union
from metagpt.actions import Action
from metagpt.logs import logger

from utils.global_vars import PROMPT_BASE_DIR


class A4_ContextHandle(Action):
    """
    A4: Convert natural language context into formal logic (FOL) premises,
    for use in downstream reasoning tasks (Step 3).
    """

    name: str = "A4ContextHandle"

    async def run(
        self,
        inputs: Union[Dict[str, Any], None] = None,
        context: str = "",
        question: str = ""
    ) -> dict:
        # Support Agent mode dictionary input
        if inputs:
            question = inputs.get("question", question)
            context = inputs.get("context", context)

        prompt_template = self._load_prompt_template()
        
        # Use .replace() to avoid KeyError when .format() handles JSON braces
        prompt = prompt_template.replace("{context}", context.strip()) \
                                .replace("{target_statement}", question.strip()) \
                                .replace("{question}", question.strip())  # Compatible with different placeholder names

        logger.info("==== [A4] Prompt to LLM ====")
        logger.debug(prompt)

        rsp = await self._aask(prompt)
        return self._parse_json_response(rsp)

    @classmethod
    def _load_prompt_template(cls) -> str:
        path = os.path.join(PROMPT_BASE_DIR, "A4_ContextHandle.txt")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"[A4] Failed to load prompt: {path} | Error: {e}")

    def _parse_json_response(self, rsp: str) -> dict:
        pattern = r"```json(.*?)```"
        match = re.search(pattern, rsp, re.DOTALL)
        try:
            json_str = match.group(1).strip() if match else rsp.strip()
            logger.debug("==== [A4] Parsed JSON ====")
            logger.debug(json_str)
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"[A4] JSON parse failed: {e}")
            return {
                "error": f"Parsing failed: {str(e)}",
                "raw_response": rsp
            }
