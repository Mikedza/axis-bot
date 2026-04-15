import asyncio
import logging
from collections import defaultdict

import discord
import yt_dlp

log = logging.getLogger("Axis")

_YDL_OPTIONS: dict = {
    "format":         "bestaudio/best",
    "quiet":          True,
    "no_warnings":    True,
    "noplaylist":     True,
    "source_address": "0.0.0.0",
}

_FFMPEG_OPTIONS: dict = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options":         "-vn",
}

class MusicHandler:
    """Manages per-guild music queues and voice client state."""

    def __init__(self) -> None:
        self.queues: dict[int, list[dict]] = defaultdict(list)
        self.voice_clients: dict[int, discord.VoiceClient] = {}

    async def fetch_track(self, query: str, requester: str) -> dict | None:
        """Search YouTube and return track metadata."""
        loop = asyncio.get_running_loop()
        try:
            with yt_dlp.YoutubeDL(_YDL_OPTIONS) as ydl:
                search_term = query if query.startswith(("http://", "https://")) else f"ytsearch1:{query}"
                info = await loop.run_in_executor(
                    None, lambda: ydl.extract_info(search_term, download=False)
                )
                
                if "entries" in info:
                    if not info["entries"]:
                        return None
                    info = info["entries"][0]

            return {
                "title":       info.get("title", "Unknown"),
                "webpage_url": info.get("webpage_url", query),
                "duration":    info.get("duration", 0),
                "requester":   requester,
            }
        except Exception as exc:
            log.error(f"yt_dlp fetch error for '{query}': {exc}")
            return None

    async def _get_stream_url(self, webpage_url: str) -> str | None:
        """Extract a fresh direct audio stream URL."""
        loop = asyncio.get_running_loop()
        try:
            with yt_dlp.YoutubeDL(_YDL_OPTIONS) as ydl:
                info = await loop.run_in_executor(
                    None, lambda: ydl.extract_info(webpage_url, download=False)
                )
            return info.get("url")
        except Exception as exc:
            log.error(f"yt_dlp stream error for '{webpage_url}': {exc}")
            return None

    async def play_next(self, guild_id: int) -> None:
        """Start playing the next track in the queue."""
        vc = self.voice_clients.get(guild_id)
        if not vc or not vc.is_connected() or vc.is_playing():
            return

        queue = self.queues[guild_id]
        if not queue:
            return

        track = queue[0]
        stream_url = await self._get_stream_url(track["webpage_url"])

        if not stream_url:
            queue.pop(0)
            await self.play_next(guild_id)
            return

        source = discord.FFmpegPCMAudio(stream_url, **_FFMPEG_OPTIONS)
        loop = asyncio.get_running_loop()

        def after_playing(error: Exception | None) -> None:
            if error:
                log.error(f"[Guild {guild_id}] Playback error: {error}")
            if self.queues[guild_id]:
                self.queues[guild_id].pop(0)
            asyncio.run_coroutine_threadsafe(self.play_next(guild_id), loop)

        vc.play(source, after=after_playing)
        log.info(f"[Guild {guild_id}] Now playing: {track['title']}")

    @staticmethod
    def format_duration(seconds: int) -> str:
        """Convert seconds to a MM:SS string."""
        m, s = divmod(seconds, 60)
        return f"{m}:{s:02d}"