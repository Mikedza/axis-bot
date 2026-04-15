import base64
import logging

import aiohttp

from config import IMAGE_GEN_ENABLED, IMAGE_GEN_URL
from personalities import (
    PERSONALITY_APPEARANCES,
    PERSONALITY_NAMES,
    SD_QUALITY_TAGS,
    SD_NEGATIVE_PROMPT,
    USER_APPEARANCE,
)
from handlers.ai_handler import AIHandler

log = logging.getLogger("Axis")


class ImageHandler:
    """Wraps Stable Diffusion WebUI image generation and AI-assisted prompt building."""

    def __init__(self, ai: AIHandler) -> None:
        self._ai = ai

    # ─── Core generation ──────────────────────────────────────────────────────
    async def generate_image(self, personality: str, extra_prompt: str = "") -> bytes | None:
        if not IMAGE_GEN_ENABLED:
            return None

        appearance = PERSONALITY_APPEARANCES.get(personality, "")
        if not appearance:
            return None

        # Combine tags: [Appearance] + [Situation] + [Quality Tags]
        parts = [p.strip() for p in [appearance, extra_prompt, SD_QUALITY_TAGS] if p.strip()]
        positive_prompt = ", ".join(parts)

        payload = {
            "prompt": positive_prompt,
            "negative_prompt": SD_NEGATIVE_PROMPT,
            "steps": 25,
            "width": 512,
            "height": 768,
            "cfg_scale": 7,
            "sampler_name": "DPM++ 2M Karras",
        }

        try:
            # Added a longer timeout because image generation is slow
            timeout = aiohttp.ClientTimeout(total=120) 
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(IMAGE_GEN_URL, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # SD returns a list of images in base64 format
                        img_bytes = base64.b64decode(data["images"][0])
                        return img_bytes
                    else:
                        log.error(f"SD WebUI Error: {resp.status}")
                        return None
        except Exception as exc:
            log.error(f"Connection to SD failed: {exc}")
            return None
    
    # ─── Situation prompt ─────────────────────────────────────────────────────

    async def build_situation_prompt(
        self, personality: str, history: str, gender: str
    ) -> str:
        """
        Use the AI to summarise a conversation into a short SD image prompt
        describing the current scene between the character and the user.
        """
        char_name = PERSONALITY_NAMES.get(personality, personality.capitalize())
        user_desc = USER_APPEARANCE.get(gender, USER_APPEARANCE["unspecified"])
        meta_prompt = (
            f"Based on this conversation between {char_name} and a user, "
            f"write a single short Stable Diffusion image prompt (max 30 words) describing what is visually happening between them. "
            f"Describe only the pose, action, and emotional mood. No dialogue. No names. "
            f"The user looks like: {user_desc}.\n\n"
            f"Conversation:\n{history.strip()[-800:]}\n\n"
            f"Image prompt:"
        )
        result = await self._ai.ask(meta_prompt, "situation-gen")
        return result.strip().split("\n")[0]  # Take only the first line.
