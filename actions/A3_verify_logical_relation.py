import re
import json
import sys
from typing import Dict, Any, Union
from metagpt.actions import Action


class A3_VerifyLogicalStructure(Action):
    name: str = "A3_VerifyLogicalStructure"

    def __init__(self, context):
        super().__init__(context=context)

    def _extract_input(self, item: Any) -> str:
        if not isinstance(item, dict):
            return str(item)
        # Full version: include both statement and FOL
        return (item.get("statement", "") + " FOL: " + item.get("FOL", "")).strip()

    async def run(self, inputs: Union[Dict[str, Any], None] = None, square: Dict[str, Any] = None) -> Dict[str, Any]:
        # Automatically support Agent unified format input
        if inputs:
            square = inputs.get("semiotic_square", {})

        # Extract proposition content (by prompt type)
        S1 = self._extract_input(square.get("S1", ""))
        S2 = self._extract_input(square.get("S2", ""))
        not_S1 = self._extract_input(square.get("not_S1", ""))
        not_S2 = self._extract_input(square.get("not_S2", ""))

        # Construct prompt
        prompt = f"""
You are a formal logic expert specializing in semantic structures and symbolic reasoning.

Your task is to determine whether the following four propositions form a valid **Greimas Semiotic Square**, which consists of two opposing concepts (A vs B) and their respective negations (¬A, ¬B).

To verify this, evaluate the logical relationships among the four propositions according to the following four criteria:
1. S1 ⊥ S2 (Contrariety)
2. S1 ⇒ ¬S2 (Mutual Exclusion)
3. S2 ⇒ ¬S1 (Symmetric Exclusion)
4. ¬S1 ∪ ¬S2 = U (Semantic Coverage)

Respond only with a JSON object:
json
{{
  "S1⊥S2": true/false,
  "S1⇒¬S2": true/false,
  "S2⇒¬S1": true/false,
  "¬S1∪¬S2=U": true/false
}}

Here are the four propositions:

S1: {S1}
S2: {S2}
¬S1: {not_S1}
¬S2: {not_S2}
"""

        rsp = await self._aask(prompt)

        match = re.search(r"\{[\s\S]*?\}", rsp)
        if match:
            try:
                checks = json.loads(match.group())
                core_keys = ["S1⊥S2", "S1⇒¬S2", "S2⇒¬S1"]
                complete_keys = core_keys + ["¬S1∪¬S2=U"]
                core_valid = all(checks.get(k) is True for k in core_keys)
                complete_valid = all(checks.get(k) is True for k in complete_keys)
                return {
                    **checks,
                    "GreimasCoreValid": core_valid,
                    "GreimasCompleteValid": complete_valid
                }
            except Exception:
                return {"GreimasCoreValid": False, "GreimasCompleteValid": False, "error": "Invalid JSON format"}
        else:
            return {"GreimasCoreValid": False, "GreimasCompleteValid": False, "error": "No structured JSON returned"}
