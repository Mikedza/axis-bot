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

    async def generate_image(
        self, personality: str, extra_prompt: str = ""
    ) -> bytes | None:
        """
        Request an image from the local SD WebUI API.

        Returns raw PNG bytes, or None if generation is disabled or fails.
        """
        if not IMAGE_GEN_ENABLED:
            log.warning("generate_image called but IMAGE_GEN_ENABLED is False.")
            return None

        appearance = PERSONALITY_APPEARANCES.get(personality, "")
        if not appearance:
            log.warning(f"No appearance defined for '{personality}'. Cannot generate image.")
            return None

        parts          = [p.strip() for p in [appearance, extra_prompt, SD_QUALITY_TAGS] if p.strip()]
        positive_prompt = ", ".join(parts)

        payload = {
            "prompt":          positive_prompt,
            "negative_prompt": SD_NEGATIVE_PROMPT,
            "steps":           28,
            "width":           512,
            "height":          768,  # Portrait ratio suits character art.
            "cfg_scale":       7,
            "sampler_name":    "DPM++ 2M Karras",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(IMAGE_GEN_URL, json=payload) as resp:
                    if resp.status == 200:
                        data      = await resp.json()
                        img_bytes = base64.b64decode(data["images"][0])
                        log.info(
                            f"Image generated for '{personality}' ({len(img_bytes)} bytes)."
                        )
                        return img_bytes
                    else:
                        log.error(f"SD WebUI returned HTTP {resp.status}.")
                        return None
        except Exception as exc:
            log.error(f"Image generation error: {exc}")
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
