from metagpt.logs import logger
import re

class TokenCaptureLoguru:
    _already_initialized = False

    def __init__(self):
        if TokenCaptureLoguru._already_initialized:
            return  # Avoid duplicate registration
        TokenCaptureLoguru._already_initialized = True
        self.records = []
        logger.add(self._sink, format="{message}", level="INFO", enqueue=True, catch=True)

    def _sink(self, message):
        match = re.search(r"prompt_tokens: (\d+), completion_tokens: (\d+)", message)
        if match:
            self.records.append({
                "prompt_tokens": int(match.group(1)),
                "completion_tokens": int(match.group(2)),
                "total_tokens": int(match.group(1)) + int(match.group(2))
            })

    def latest(self):
        return self.records[-1] if self.records else {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    def summary(self):
        prompt_total = sum(r["prompt_tokens"] for r in self.records)
        completion_total = sum(r["completion_tokens"] for r in self.records)
        return {
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
            "total_tokens": prompt_total + completion_total
        }

    def clear(self):
        self.records.clear()
