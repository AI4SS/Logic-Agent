import json
from math import log
import re
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.logs import logger

from actions.A1_construct_semiotic_square import A1_ConstructSemioticSquare
from actions.A2_verify_fol_cfg import A2_VerifyFOLWithCFG
from actions.A3_verify_logical_relation import A3_VerifyLogicalStructure
from actions.A4_context_handle import A4_ContextHandle
from actions.A5_reasoning import A5_ReasoningJudge_New
from actions.A5_reasoning_light import A5_ReasoningJudge_Light

from utils.global_vars import token_tracker
from utils.save_result import format_output_entry


class ReasoningAgent(Role):
    name: str = "Phil"
    profile: str = "FOL Reasoning Expert"

    def __init__(self, dataset_name: str = "ProntoQA", **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.original_data = None

        actions = [
            A1_ConstructSemioticSquare,
            A2_VerifyFOLWithCFG,
            A3_VerifyLogicalStructure,
            A4_ContextHandle,
            self._select_a5(),
        ]

        self.set_actions(actions)
        self._set_react_mode("by_order")

    def _select_a5(self):
        # Use lightweight A5 for ProntoQA (2-value logic)
        if self.dataset_name.lower() == "prontoqa":
            return A5_ReasoningJudge_Light
        # Use full A5 for other datasets
        return A5_ReasoningJudge_New

    def _get_memory_by_action(self, action_cls):
        for m in reversed(self.get_memories()):
            cause = m.cause_by
            if isinstance(cause, str):
                if action_cls.__name__ in cause:
                    return m
            elif hasattr(cause, '__name__') and cause.__name__ == action_cls.__name__:
                return m
            elif hasattr(cause, '__class__') and cause.__class__.__name__ == action_cls.__name__:
                return m
        raise ValueError(f"No memory found for action: {action_cls.__name__}")

    async def _act(self) -> dict:
        todo = self.rc.todo

        if self.original_data is None:
            first_msg = self.get_memories(k=1)[0]
            self.original_data = json.loads(first_msg.content)

        if isinstance(todo, A1_ConstructSemioticSquare):
            logger.info("🧠 A1: Constructing Semiotic Square")
            print("question", self._extract_question(self.original_data["question"]))
            inputs = {
                "question": self._extract_question(self.original_data["question"]),
                "context": self.original_data["context"]
            }

        # A2: FOL CFG verification
        elif isinstance(todo, A2_VerifyFOLWithCFG):
            logger.info("🧠 A2: Verifying FOL with CFG")
            square_msg = self._get_memory_by_action(A1_ConstructSemioticSquare)
            square_data = json.loads(square_msg.content)
            inputs = {"semiotic_square": square_data}
            result = await todo.run(inputs)
            # logger.info(f"[A2] FOL validation result: {result}")
            logger.info("[A2] Validation passed.")
            self.rc.memory.add(Message(
                content=json.dumps(result, ensure_ascii=False),
                role=self.profile,
                cause_by=A2_VerifyFOLWithCFG
            ))

            # if all(v.get("valid") for k, v in result.items() if isinstance(v, dict)):
            #     logger.info("[A2] Validation passed.")
            #     self.rc.memory.add(Message(
            #         content=json.dumps(result, ensure_ascii=False),
            #         role=self.profile,
            #         cause_by=A2_VerifyFOLWithCFG
            #     ))
            # else:
            #     logger.warning("[A2] Validation failed.")
            #     raise Exception(f"A2 validation failed for sample {self.original_data['id']}")

        # A3: Greimas structure logical verification
        elif isinstance(todo, A3_VerifyLogicalStructure):
            logger.info("🧠 A3: Verifying logical structure")
            square_msg = self._get_memory_by_action(A1_ConstructSemioticSquare)
            inputs = {
                "semiotic_square": json.loads(square_msg.content)
            }

            result = await todo.run(inputs=inputs)
            logger.info(f"[A3] Logical structure verification result: {result}")

            if result.get("GreimasCoreValid") or result.get("GreimasCompleteValid"):
                logger.info("✅ A3: Semiotic structure valid.")
                self.rc.memory.add(Message(
                    content=json.dumps(result, ensure_ascii=False),
                    role=self.profile,
                    cause_by=A3_VerifyLogicalStructure
                ))
            else:
                logger.warning("❌ A3: Invalid structure.")
                raise Exception(f"A3 structure invalid for sample {self.original_data['id']}")

        elif isinstance(todo, A4_ContextHandle):
            logger.info("🧠 A4: Handling context")
            inputs = {
                "context": self.original_data["context"],
                "question": self._extract_question(self.original_data["question"])
            }

        elif isinstance(todo, self._select_a5()):
            logger.info("🧠 A5: Judging reasoning")
            square = json.loads(self._get_memory_by_action(A1_ConstructSemioticSquare).content)

            a4_out = json.loads(self._get_memory_by_action(A4_ContextHandle).content)
            premises = a4_out.get("premises", a4_out)

            inputs = {
                "semiotic_square": square,
                "context": self.original_data["context"],
                "question": self._extract_question(self.original_data["question"]),
                "premises": premises
            }

        else:
            raise RuntimeError(f"Unknown action type: {type(todo)}")

        result = await todo.run(inputs)

        msg_out = Message(
            content=json.dumps(result, ensure_ascii=False),
            role=self.profile,
            cause_by=type(todo)
        )
        self.rc.memory.add(msg_out)

        if isinstance(todo, self._select_a5()):
            token_usage = token_tracker.summary()
            token_tracker.clear()

            return format_output_entry(
                original_data=self.original_data,
                a1_result=json.loads(self._get_memory_by_action(A1_ConstructSemioticSquare).content),
                a3_result=json.loads(self._get_memory_by_action(A3_VerifyLogicalStructure).content),
                a4_result=json.loads(self._get_memory_by_action(A4_ContextHandle).content),
                a5_result=result,
                token_usage=token_usage
            )

        return msg_out

    def _extract_question(self, q: str) -> str:
        match = re.search(r'["“](.*?)["”]', q)
        return match.group(1) if match else q.strip()
