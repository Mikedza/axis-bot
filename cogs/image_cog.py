import io
import logging

import discord
from discord import app_commands
from discord.ext import commands

from config import IMAGE_GEN_ENABLED
from handlers.database import Database
from handlers.image_handler import ImageHandler
from personalities import PERSONALITY_NAMES

log = logging.getLogger("Axis")


class ImageCog(commands.Cog):
    """Slash commands for Stable Diffusion image generation."""

    def __init__(self, bot: commands.Bot, db: Database, image: ImageHandler) -> None:
        self.bot   = bot
        self.db    = db
        self.image = image

    # ─── /imagine ─────────────────────────────────────────────────────────────

    @app_commands.command(
        name="imagine",
        description="Generate an image of your companion (requires IMAGE_GEN_ENABLED)",
    )
    async def imagine(self, interaction: discord.Interaction) -> None:
        if not IMAGE_GEN_ENABLED:
            await interaction.response.send_message(
                "Image generation is not enabled yet. Check back later! 🎨", ephemeral=True
            )
            return

        user = await self.db.get_user(str(interaction.user.id))
        if not user:
            await interaction.response.send_message(
                "Please use `/start-axis` first!", ephemeral=True
            )
            return

        await interaction.response.defer()

        personality = self.db.row_get(user, 1, "gym")
        img_bytes   = await self.image.generate_image(personality)

        if not img_bytes:
            await interaction.followup.send(
                "Image generation failed. Check that SD WebUI is running."
            )
            return

        file         = discord.File(fp=io.BytesIO(img_bytes), filename="companion.png")
        display_name = PERSONALITY_NAMES.get(personality, personality.capitalize())
        await interaction.followup.send(content=f"Here's **{display_name}**! 🎨", file=file)
        log.info(f"[{interaction.user}] Imagine generated for '{personality}'.")

    # ─── /generate-situation ──────────────────────────────────────────────────

    @app_commands.command(
        name="generate-situation",
        description="Generate an image of your companion based on your current conversation",
    )
    async def generate_situation(self, interaction: discord.Interaction) -> None:
        if not IMAGE_GEN_ENABLED:
            await interaction.response.send_message(
                "Image generation is not enabled yet. Check back later! 🎨", ephemeral=True
            )
            return

        user = await self.db.get_user(str(interaction.user.id))
        if not user:
            await interaction.response.send_message(
                "Please use `/start-axis` first!", ephemeral=True
            )
            return

        personality = self.db.row_get(user, 1, "gym")
        memories    = self.db.parse_memories(user[2] if len(user) > 2 else None)
        history     = memories.get(personality, "").strip()
        gender      = self.db.row_gender(user)

        if not history:
            await interaction.response.send_message(
                "No conversation history yet — chat a bit first, then use this command! 💬",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        situation_prompt = await self.image.build_situation_prompt(personality, history, gender)
        log.info(f"[{interaction.user}] Situation prompt: {situation_prompt}")

        img_bytes = await self.image.generate_image(personality, extra_prompt=situation_prompt)

        if not img_bytes:
            await interaction.followup.send(
                "Image generation failed. Check that SD WebUI is running."
            )
            return

        file         = discord.File(fp=io.BytesIO(img_bytes), filename="situation.png")
        display_name = PERSONALITY_NAMES.get(personality, personality.capitalize())
        await interaction.followup.send(
            content=f"**{display_name}** — _{situation_prompt}_ 🎨", file=file
        )
        log.info(f"[{interaction.user}] Situation image generated for '{personality}'.")
