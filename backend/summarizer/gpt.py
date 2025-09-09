import os
from typing import List
from dotenv import load_dotenv


def summarize_captions(frame_texts: List[str], target_pages: int) -> List[str]:
    """Placeholder for GPT-based summarization; returns trimmed or padded list."""
    load_dotenv()
    # In a full implementation, call OpenAI's Chat Completions to generate concise, emotional captions.
    # For now, just return the inputs or placeholders if empty.
    if not frame_texts:
        return ["..."] * target_pages
    return frame_texts[:target_pages] + ["..."] * max(0, target_pages - len(frame_texts))

