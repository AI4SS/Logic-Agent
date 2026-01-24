from typing import Dict, Any, List, Optional, Union
from metagpt.actions import Action
from tools.fol_parser import parse_text_FOL_to_tree

class A2_VerifyFOLWithCFG(Action):
    name: str = "A2_VerifyFOLWithCFG"

    async def run(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        semiotic_square: Optional[Dict[str, Any]] = None,
        premises: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        # Automatically support MetaGPT agent input structure
        if inputs:
            semiotic_square = inputs.get("semiotic_square", semiotic_square)
            premises = inputs.get("premises", premises)

        if premises:
            return self._verify_premises(premises)
        elif semiotic_square:
            return self._verify_square(semiotic_square)
        else:
            raise ValueError("Must provide either `semiotic_square` or `premises`.")

    def _verify_square(self, square: Dict[str, Any]) -> Dict[str, Any]:
        results = {}

        # Map Unicode key names to standard key names
        key_aliases = {
            "¬S1": "not_S1",
            "¬S2": "not_S2"
        }
        # Copy original square to avoid modifying input
        square = {**square}

        for unicode_key, ascii_key in key_aliases.items():
            if unicode_key in square and ascii_key not in square:
                square[ascii_key] = square[unicode_key]

        for key in ["S1", "S2", "not_S1", "not_S2"]:
            item = square.get(key)
            fol_expr = None

            if isinstance(item, dict) and "FOL" in item:
                fol_expr = item["FOL"]
            elif isinstance(item, str):
                fol_expr = item

            if fol_expr:
                if self._should_skip_fol_validation(fol_expr):
                    results[key] = {
                        "fol": fol_expr,
                        "valid": True,
                        "skipped": True,
                        "reason": "Simple atomic predicate, no logic symbols"
                    }
                    continue

                try:
                    tree = parse_text_FOL_to_tree(fol_expr)
                    results[key] = {
                        "fol": fol_expr,
                        "valid": tree is not None
                    }
                except Exception as e:
                    results[key] = {
                        "fol": fol_expr,
                        "valid": False,
                        "error": str(e)
                    }
            else:
                results[key] = {
                    "skipped": True,
                    "reason": "No FOL expression found"
                }

        return results


    def _should_skip_fol_validation(self, s: str) -> bool:
        return not any(sym in s for sym in ["∀", "∃", "→", "∧", "¬", "∨", "↔"])


    def _verify_premises(self, premises: List[str]) -> Dict[str, Any]:
        results = {}
        for idx, fol_expr in enumerate(premises):
            try:
                tree = parse_text_FOL_to_tree(fol_expr)
                results[f"premise_{idx + 1}"] = {
                    "fol": fol_expr,
                    "valid": tree is not None
                }
            except Exception as e:
                results[f"premise_{idx + 1}"] = {
                    "fol": fol_expr,
                    "valid": False,
                    "error": str(e)
                }
        return results
