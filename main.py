import logging

import discord
from discord.ext import commands

from config import (
    TOKEN,
    MAX_CONCURRENT,
    REQUEST_TIMEOUT,
    MEMORY_LIMIT,
    IMAGE_GEN_ENABLED,
)
from handlers.ai_handler import AIHandler
from handlers.database import Database
from handlers.image_handler import ImageHandler
from handlers.music_handler import MusicHandler

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("axis.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("Axis")

# ─── Intents ──────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.voice_states = True  # Required to detect which voice channel the user is in.


# ─── Bot ──────────────────────────────────────────────────────────────────────
class AxisBot(commands.Bot):
    """
    Top-level bot class.

    Service objects are created here and injected into each cog so that every
    component gets exactly the instance it needs — no global state required.
    """

    def __init__(self) -> None:
        # Disable the built-in prefix-based help command to avoid a name clash
        # with our /help slash command registered in FunCog.
        super().__init__(command_prefix="!", help_command=None, intents=intents)

        # ── Services ──────────────────────────────────────────────────────────
        self.db    = Database()
        self.ai    = AIHandler(MAX_CONCURRENT, REQUEST_TIMEOUT)
        self.music = MusicHandler()
        self.image = ImageHandler(self.ai)

    async def setup_hook(self) -> None:
        """
        Called once by discord.py before the bot connects.
        Initialise the database, register cogs, and sync slash commands.
        """
        await self.db.init()

        # Import here to avoid circular imports at module load time.
        from cogs.ai_cog    import AICog
        from cogs.fun_cog   import FunCog
        from cogs.image_cog import ImageCog
        from cogs.music_cog import MusicCog

        await self.add_cog(AICog(self, self.db, self.ai))
        await self.add_cog(ImageCog(self, self.db, self.image))
        await self.add_cog(MusicCog(self, self.music))
        await self.add_cog(FunCog(self, self.db))

        # --- INSTANT SYNC FOR DEVELOPMENT ---
        # Replace 1234567890 with your actual Server ID
        # MY_GUILD = discord.Object(id=1484235279547105583)
        # self.tree.clear_commands(guild=MY_GUILD)
        # await self.tree.sync(guild=MY_GUILD)
        # await self.tree.sync()

        await self.tree.sync()
        log.info("Slash commands synced.")

    async def on_ready(self) -> None:
        await self.change_presence(
            status=discord.Status.online,
            activity=discord.Activity(
                type=discord.ActivityType.playing, name="with ya 💕"
            ),
        )
        log.info(
            f"Axis is online as {self.user}  |  "
            f"concurrency={MAX_CONCURRENT}  |  "
            f"timeout={REQUEST_TIMEOUT}s  |  "
            f"memory_limit={MEMORY_LIMIT} exchanges/personality  |  "
            f"image_gen={'ON' if IMAGE_GEN_ENABLED else 'OFF (scaffolded)'}"
        )


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    AxisBot().run(TOKEN)
