"""Writer message templates and input batches."""

import copy
from types import SimpleNamespace

from memory import writer_prompt


class MemoryProcessor:
    """Processor proxy with writer message text."""

    def __init__(self, original, prompt):
        """Store the wrapped processor and writer prompt."""
        self.original, self.prompt = original, prompt

    def __getattr__(self, name):
        """Return the named attribute from the wrapped processor."""
        return getattr(self.original, name)

    def __call__(self, *args, **kwargs):
        """Return the wrapped processor output for the arguments."""
        return self.original(*args, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        """Return rendered messages with writer text inserted."""
        messages = copy.deepcopy(messages)
        messages[0]["content"][0]["text"] = self.prompt
        messages[0]["content"][-1]["text"] = (
            "Current chunk ends here. Output the complete replacement memory."
        )
        self.messages = messages
        return self.original.apply_chat_template(messages, **kwargs)


def memory_batch(route, row, call, old_memory):
    """Encode row query, call frames, and old_memory with route.

    Return model inputs, prefix length, and input metadata.
    """
    from multimodal import batch_for

    text = writer_prompt(row["raw_query"], call["real_indices"], old_memory)
    processor = MemoryProcessor(route.processor, text)
    view = SimpleNamespace(qwen=route.qwen, processor=processor)
    chunk = dict(
        paths=call["paths"],
        observation_ids=call["real_indices"],
        source_indices=call["source_indices"],
        times_seconds=call["real_indices"],
        units="INDEX",
    )
    data, prefix, audit = batch_for(view, chunk)
    audit.update(
        memory_in=old_memory,
        prompt=text,
        messages=processor.messages,
        GT_in_input=False,
        language_KV_reused=False,
        real_indices=call["real_indices"],
        target_index=call["target_index"],
        rendered_prompt=route.processor.tokenizer.decode(
            data["input_ids"][0, :prefix], skip_special_tokens=False
        ),
    )
    return data, prefix, audit
