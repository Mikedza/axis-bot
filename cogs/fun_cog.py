import logging

import discord
from discord import app_commands
from discord.ext import commands

from handlers.database import Database
from personalities import PERSONALITY_NAMES

log = logging.getLogger("Axis")


class FunCog(commands.Cog):
    """Leaderboard and help commands."""

    def __init__(self, bot: commands.Bot, db: Database) -> None:
        self.bot = bot
        self.db  = db

    # ─── /popularity ──────────────────────────────────────────────────────────

    @app_commands.command(
        name="popularity",
        description="See who has chatted with Axis the most",
    )
    async def popularity(self, interaction: discord.Interaction) -> None:
        rows = await self.db.get_top_chatters(10)

        if not rows:
            await interaction.response.send_message(
                "No chat history yet. Start talking with `/say`!", ephemeral=True
            )
            return

        medals = ["🥇", "🥈", "🥉"]
        embed  = discord.Embed(title="🏆 Top Axis Chatters", color=discord.Color.gold())

        for i, (user_id, personality, count) in enumerate(rows):
            medal    = medals[i] if i < 3 else f"#{i + 1}"
            member   = interaction.guild.get_member(int(user_id)) if interaction.guild else None
            display  = member.display_name if member else f"User {user_id}"
            companion = PERSONALITY_NAMES.get(personality, personality.capitalize())
            embed.add_field(
                name=f"{medal} {display}",
                value=f"Companion: **{companion}** | Messages: **{count}**",
                inline=False,
            )
        await interaction.response.send_message(embed=embed)

    # ─── /help ────────────────────────────────────────────────────────────────

    @app_commands.command(name="help", description="See all available Axis commands")
    async def help_command(self, interaction: discord.Interaction) -> None:
        embed = discord.Embed(
            title="Axis — Command Guide",
            description="Everything I can do, all in one place.",
            color=discord.Color.blurple(),
        )

        embed.add_field(name="🤖  AI Companion", value="\u200b", inline=False)
        embed.add_field(name="`/start-axis`",  value="Set up your AI companion (personality → gender → name).", inline=False)
        embed.add_field(name="`/clear-axis`",  value="Permanently wipe all your memories and start fresh.",      inline=False)
        embed.add_field(name="`/status-axis`", value="View your profile — companion, mood, memory, and more.",   inline=False)
        embed.add_field(name="`/setname`",     value="Tell your companion your name so they never forget.",      inline=False)
        embed.add_field(name="`/set-gender`",  value="Update your gender so your companion addresses you correctly.", inline=False)
        embed.add_field(name="`/say <message>`",     value="Talk to your companion publicly in the channel.",   inline=False)
        embed.add_field(name="`/whisper <message>`", value="Talk privately — response arrives via DM.",         inline=False)
        embed.add_field(name="`/history`",     value="View your full conversation history with your current companion.", inline=False)

        embed.add_field(name="🎨  Image Generation", value="\u200b", inline=False)
        embed.add_field(name="`/imagine`",            value="Generate an image of your companion.",                        inline=False)
        embed.add_field(name="`/generate-situation`", value="Generate a scene based on your current conversation.",        inline=False)

        embed.add_field(name="🎵  Music", value="\u200b", inline=False)
        embed.add_field(name="`/play <query or URL>`", value="Play a song by name or YouTube link. Queues automatically.", inline=False)
        embed.add_field(name="`/play-queue`",          value="Show all songs coming up in the queue.",                     inline=False)
        embed.add_field(name="`/skip`",                value="Skip the current song.",                                     inline=False)
        embed.add_field(name="`/leave-call`",          value="Make Axis leave the voice channel and clear the queue.",     inline=False)
        embed.add_field(name="`/reset-queue`",         value="Clear all songs without disconnecting.",                     inline=False)

        embed.add_field(name="🎉  Fun", value="\u200b", inline=False)
        embed.add_field(name="`/popularity`", value="Leaderboard of who has chatted with Axis the most.", inline=False)

        embed.set_footer(text="Use /start-axis to get started!")
        await interaction.response.send_message(embed=embed, ephemeral=True)
