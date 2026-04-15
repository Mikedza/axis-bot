import asyncio
import logging

import aiohttp

from config import OLLAMA_URL, OLLAMA_MODEL
from personalities import PERSONALITIES

log = logging.getLogger("Axis")


class AIHandler:
    """Wraps all communication with the local Ollama instance."""

    def __init__(self, max_concurrent: int, request_timeout: int) -> None:
        self._semaphore    = asyncio.Semaphore(max_concurrent)
        self._timeout      = request_timeout
        self.queue_depth   = 0  # Informational counter; safe in asyncio single-thread.

    # ─── Low-level request ────────────────────────────────────────────────────

    async def ask(self, prompt: str, user_tag: str) -> str:
        """Send *prompt* to Ollama and return the response text."""
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        log.info(f"[{user_tag}] Sending prompt ({len(prompt)} chars) to Ollama …")
        loop    = asyncio.get_running_loop()
        t_start = loop.time()

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(OLLAMA_URL, json=payload) as resp:
                    elapsed = loop.time() - t_start
                    if resp.status == 200:
                        data  = await resp.json()
                        reply = data.get("response", "").strip()
                        log.info(
                            f"[{user_tag}] Ollama replied in {elapsed:.1f}s "
                            f"({len(reply)} chars)."
                        )
                        return reply or "System Error: Received an empty response from the AI."
                    else:
                        log.warning(
                            f"[{user_tag}] Ollama HTTP {resp.status} after {elapsed:.1f}s."
                        )
                        return f"System Error: AI service returned status {resp.status}."

        except asyncio.TimeoutError:
            log.error(f"[{user_tag}] Ollama timed out after {self._timeout}s.")
            return "System Error: The AI took too long to respond. Please try again."
        except aiohttp.ClientConnectorError as exc:
            log.error(f"[{user_tag}] Ollama connection refused: {exc}")
            return "System Error: Cannot reach Ollama. Is it running on localhost:11434?"
        except Exception as exc:
            log.error(f"[{user_tag}] Unexpected Ollama error: {exc}")
            return f"System Error: Unexpected failure ({type(exc).__name__})."

    # ─── Throttled request ────────────────────────────────────────────────────

    async def ask_throttled(self, prompt: str, user_tag: str) -> str:
        """Same as ``ask`` but honours the concurrency semaphore."""
        self.queue_depth += 1
        log.info(f"[{user_tag}] Queued (depth={self.queue_depth})")
        try:
            async with self._semaphore:
                return await self.ask(prompt, user_tag)
        finally:
            self.queue_depth -= 1

    # ─── Prompt builder ───────────────────────────────────────────────────────

    @staticmethod
    def build_prompt(
        personality: str,
        traits: str,
        mood: str,
        history: str,
        user_message: str,
        user_name: str = "",
        gender: str = "unspecified",
    ) -> str:
        """Assemble the full prompt string sent to the AI model."""
        history_block = (
            f"[Conversation History]\n{history.strip()}\n\n"
            if history.strip()
            else ""
        )
        name_line = (
            f"User's Name: {user_name} — always address them by this exact name, remember it permanently.\n"
            if user_name
            else "User's Name: Unknown — learn it naturally if they share it, then remember it permanently.\n"
        )
        gender_line = (
            f"User's Gender: {gender} — always use the correct pronouns for this person.\n"
            if gender != "unspecified"
            else ""
        )
        return (
            f"[System Instruction]\n{PERSONALITIES[personality]}\n\n"
            f"[User Profile]\n"
            f"{name_line}"
            f"{gender_line}"
            f"Traits: {traits or 'Unknown'}\n"
            f"Current Mood: {mood or 'Neutral'}\n\n"
            f"{history_block}"
            f"[Current Message]\n"
            f"User: {user_message}\n"
            f"Assistant:"
        )
