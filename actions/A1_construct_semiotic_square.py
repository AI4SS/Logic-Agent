# PhilAgent/actions/A1_construct_semiotic_square.py

import re
import json
import sys
from typing import ClassVar, Optional, Union
from metagpt.context import Context
from metagpt.actions import Action
from metagpt.logs import logger
import os

from utils.global_vars import PROMPT_BASE_DIR

class A1_ConstructSemioticSquare(Action):
    """
    Step 1: Extract opposing concepts and construct the Semiotic Square.
    Supports both manual and agent-style execution.
    """

    name: str = "A1_ConstructSemioticSquare"

    def __init__(self, context: Context):
        super().__init__(context=context)
        self.prompt_file = os.path.join(PROMPT_BASE_DIR, "A1_ConstructSemioticSquare.txt")


    async def run(
        self,
        question: Union[str, dict],
        context: Optional[str] = None
    ):
        # Determine if input is Agent dict (new version) or user-provided parameters (old version)
        if isinstance(question, dict):
            logger.debug("[A1] Running in Agent mode (dict input)")
            inputs = question
            question = inputs.get("question", "")
            context = inputs.get("context", "")
        else:
            logger.debug("[A1] Running in manual mode (str input)")
            question = question or ""
            context = context or ""

        # Construct prompt
        prompt_template = self._load_prompt_template()
        prompt = prompt_template.format(question=question, context=context)

        logger.info("==== [A1] Prompt to LLM ====")
        logger.debug(prompt)

        rsp = await self._aask(prompt)
        square = A1_ConstructSemioticSquare.parse_json(rsp)

        logger.info("==== [A1] Parsed Semiotic Square ====")
        logger.info(json.dumps(square, indent=2, ensure_ascii=False))

        return square


    def _load_prompt_template(self) -> str:
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt template: {e}")

    @staticmethod
    def parse_json(rsp):
        if isinstance(rsp, dict):
            return rsp
        
        pattern = r"```json(.*?)```"
        match = re.search(pattern, str(rsp), re.DOTALL)
        try:
            json_str = match.group(1).strip() if match else rsp.strip()

            # Auto-fix trailing commas: remove all `,` followed by `}` or `]`
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

            logger.debug("=== [A1] Extracted JSON Payload ===")
            logger.debug(json_str)
            raw = json.loads(json_str)

            # Compatible with nested format { "Semiotic Square": { ... } }
            square = raw.get("Semiotic Square", raw)

            # Map unicode keys (e.g. "¬S1") to standard ASCII keys (e.g. "not_S1")
            key_mapping = {
                "¬S1": "not_S1",
                "¬S2": "not_S2"
            }
            for unicode_key, ascii_key in key_mapping.items():
                if unicode_key in square and ascii_key not in square:
                    square[ascii_key] = square[unicode_key]

            # Finally, keep only target structure fields
            result = {
                "S1": square.get("S1", {}),
                "S2": square.get("S2", {}),
                "not_S1": square.get("not_S1", {}),
                "not_S2": square.get("not_S2", {}),
                "S2_type": square.get("S2_type", "")
            }

            return result

        except Exception as e:
            logger.error(f"=== [A1] JSON Parsing Failed: {e} ===")
            return {
                "error": f"Parsing failed: {str(e)}",
                "raw_response": rsp
            }

