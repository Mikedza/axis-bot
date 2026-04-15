import logging

import discord
from discord import app_commands
from discord.ext import commands

from config import MEMORY_LIMIT, DISCORD_MAX_LEN
from handlers.ai_handler import AIHandler
from handlers.database import Database
from personalities import PERSONALITY_NAMES, PERSONALITY_DESCRIPTIONS
from ui.modals import NameModal
from ui.views import ConfirmClearView, GenderView, PersonalityView

log = logging.getLogger("Axis")


class AICog(commands.Cog):
    """All AI-companion slash commands and the shared chat pipeline."""

    def __init__(self, bot: commands.Bot, db: Database, ai: AIHandler) -> None:
        self.bot = bot
        self.db  = db
        self.ai  = ai

    # ─── /start-axis ──────────────────────────────────────────────────────────

    @app_commands.command(
        name="start-axis",
        description="Set up your Axis AI companion (3-step flow)",
    )
    async def start_axis(self, interaction: discord.Interaction) -> None:
        if await self.db.user_exists(str(interaction.user.id)):
            await interaction.response.send_message(
                "You already have an Axis profile. Use `/clear-axis` to start over.",
                ephemeral=True,
            )
            return

        log.info(f"[{interaction.user}] Starting Axis setup.")

        embed = discord.Embed(
            title="Step 1 — Choose Your Companion",
            description="Pick who you want to talk to. Use the dropdown below.",
            color=discord.Color.blurple(),
        )
        for key, name in PERSONALITY_NAMES.items():
            embed.add_field(
                name=f"{name}  ({key.capitalize()})",
                value=PERSONALITY_DESCRIPTIONS[key],
                inline=False,
            )

        await interaction.response.send_message(
            embed=embed,
            view=PersonalityView(interaction.user.id, self.db),
            ephemeral=True,
        )

    # ─── /set-gender ──────────────────────────────────────────────────────────

    @app_commands.command(
        name="set-gender",
        description="Update your gender so your companion addresses you correctly",
    )
    async def set_gender(self, interaction: discord.Interaction) -> None:
        if not await self.db.user_exists(str(interaction.user.id)):
            await interaction.response.send_message(
                "Please use `/start-axis` first!", ephemeral=True
            )
            return
        await interaction.response.send_message(
            "Select your gender:",
            view=GenderView(interaction.user.id, self.db, from_setup=False),
            ephemeral=True,
        )

    # ─── /clear-axis ──────────────────────────────────────────────────────────

    @app_commands.command(
        name="clear-axis",
        description="Permanently delete your Axis memory",
    )
    async def clear_axis(self, interaction: discord.Interaction) -> None:
        if not await self.db.user_exists(str(interaction.user.id)):
            await interaction.response.send_message(
                "You don't have an Axis profile to clear.", ephemeral=True
            )
            return
        log.info(f"[{interaction.user}] Requested memory clear.")
        await interaction.response.send_message(
            "This will permanently delete ALL your Axis memories across every personality. "
            "Are you sure?",
            view=ConfirmClearView(self.db),
            ephemeral=True,
        )

    # ─── /status-axis ─────────────────────────────────────────────────────────

    @app_commands.command(
        name="status-axis",
        description="View your current Axis profile",
    )
    async def status_axis(self, interaction: discord.Interaction) -> None:
        user = await self.db.get_user(str(interaction.user.id))
        if not user:
            await interaction.response.send_message(
                "No profile found. Use `/start-axis` to get started.", ephemeral=True
            )
            return

        personality   = self.db.row_get(user, 1, "gym")
        memories      = self.db.parse_memories(user[2] if len(user) > 2 else None)
        traits        = self.db.row_get(user, 3, "None observed yet")
        mood          = self.db.row_get(user, 4, "Neutral")
        user_name     = self.db.row_get(user, 5, "Not set — use /setname")
        prompt_count  = int(user[6]) if len(user) > 6 and user[6] else 0
        gender        = self.db.row_gender(user)
        history       = memories.get(personality, "")
        exchange_count = len([l for l in history.split("\n") if l.strip()]) // 2
        display_name  = PERSONALITY_NAMES.get(personality, personality.capitalize())

        embed = discord.Embed(title="Your Axis Profile", color=discord.Color.blurple())
        embed.add_field(name="Companion",       value=f"{display_name} ({personality.capitalize()})", inline=True)
        embed.add_field(name="Current Mood",    value=mood,                  inline=True)
        embed.add_field(name="Your Name",       value=user_name,             inline=True)
        embed.add_field(name="Gender",          value=gender.capitalize(),   inline=True)
        embed.add_field(name="Observed Traits", value=traits or "None observed yet", inline=False)
        embed.add_field(name="Memory",          value=f"{exchange_count}/{MEMORY_LIMIT} exchanges stored", inline=True)
        embed.add_field(name="Total Prompts",   value=str(prompt_count),     inline=True)
        embed.set_footer(
            text="/clear-axis to reset  |  /setname to update name  |  /set-gender to update gender"
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # ─── /setname ─────────────────────────────────────────────────────────────

    @app_commands.command(
        name="setname",
        description="Tell Axis your name so it never forgets",
    )
    async def setname(self, interaction: discord.Interaction) -> None:
        if not await self.db.user_exists(str(interaction.user.id)):
            await interaction.response.send_message(
                "Please use `/start-axis` first!", ephemeral=True
            )
            return
        await interaction.response.send_modal(
            NameModal(interaction.user.id, self.db, from_setup=False)
        )

    # ─── /say ─────────────────────────────────────────────────────────────────

    @app_commands.command(name="say", description="Talk to Axis publicly in the channel")
    async def say(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._handle_chat(interaction, prompt, private=False)

    # ─── /whisper ─────────────────────────────────────────────────────────────

    @app_commands.command(name="whisper", description="Talk to Axis privately via DM")
    async def whisper(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._handle_chat(interaction, prompt, private=True)

    # ─── /history ─────────────────────────────────────────────────────────────

    @app_commands.command(
        name="history",
        description="View your full conversation history with Axis",
    )
    async def history_cmd(self, interaction: discord.Interaction) -> None:
        user = await self.db.get_user(str(interaction.user.id))
        if not user:
            await interaction.response.send_message(
                "No profile found. Use `/start-axis` to get started.", ephemeral=True
            )
            return

        personality  = self.db.row_get(user, 1, "gym")
        memories     = self.db.parse_memories(user[2] if len(user) > 2 else None)
        raw_history  = memories.get(personality, "").strip()
        display_name = PERSONALITY_NAMES.get(personality, personality.capitalize())

        if not raw_history:
            await interaction.response.send_message(
                f"No conversation history yet for **{display_name}**. "
                f"Use `/say` to start chatting!",
                ephemeral=True,
            )
            return

        lines           = [l for l in raw_history.split("\n") if l.strip()]
        formatted_lines: list[str] = []
        exchange_num    = 1
        i               = 0

        while i < len(lines):
            user_line = lines[i]
            bot_line  = lines[i + 1] if i + 1 < len(lines) else ""
            formatted_lines.append(f"**[{exchange_num}]**")
            formatted_lines.append(f"🧑 {user_line}")
            if bot_line:
                formatted_lines.append(f"🤖 {bot_line}")
            formatted_lines.append("")  # Blank separator between exchanges.
            exchange_num += 1
            i += 2

        header        = f"📜 **Conversation history with {display_name}** ({exchange_num - 1} exchanges)\n\n"
        chunks: list[str] = []
        current_chunk = header

        for line in formatted_lines:
            candidate = current_chunk + line + "\n"
            if len(candidate) > DISCORD_MAX_LEN:
                chunks.append(current_chunk.rstrip())
                current_chunk = line + "\n"
            else:
                current_chunk = candidate
        if current_chunk.strip():
            chunks.append(current_chunk.rstrip())

        await interaction.response.send_message(chunks[0], ephemeral=True)
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk, ephemeral=True)

        log.info(
            f"[{interaction.user}] History viewed ({exchange_num - 1} exchanges, {personality})."
        )

    # ─── Core chat pipeline ───────────────────────────────────────────────────

    async def _handle_chat(
        self, interaction: discord.Interaction, prompt: str, private: bool = False
    ) -> None:
        """Process a message through the AI and deliver the response."""
        user_tag = str(interaction.user)

        try:
            await interaction.response.defer(ephemeral=private)

            user = await self.db.get_user(str(interaction.user.id))
            if not user:
                await interaction.followup.send(
                    "Please use `/start-axis` first!", ephemeral=True
                )
                return

            personality  = self.db.row_get(user, 1, "gym")
            memories     = self.db.parse_memories(user[2] if len(user) > 2 else None)
            traits       = self.db.row_get(user, 3, "")
            mood         = self.db.row_get(user, 4, "Neutral")
            user_name    = self.db.row_get(user, 5, "")
            prompt_count = (int(user[6]) if len(user) > 6 and user[6] else 0) + 1
            gender       = self.db.row_gender(user)
            history      = memories.get(personality, "")

            full_prompt = AIHandler.build_prompt(
                personality, traits, mood, history, prompt, user_name, gender
            )
            response = await self.ai.ask_throttled(full_prompt, user_tag)

            new_history         = history + f"\nUser: {prompt}\nAssistant: {response}"
            memories[personality] = self.db.trim_history(new_history, MEMORY_LIMIT)
            await self.db.save_user(
                str(interaction.user.id),
                personality, memories, traits, mood, user_name, prompt_count, gender,
            )

            safe_response = (
                response[:DISCORD_MAX_LEN] + "…"
                if len(response) > DISCORD_MAX_LEN
                else response
            )

            if private:
                try:
                    dm = await interaction.user.create_dm()
                    await dm.send(safe_response)
                    await interaction.followup.send("Check your DMs!", ephemeral=True)
                except discord.Forbidden:
                    await interaction.followup.send(
                        "I can't DM you. Enable DMs from server members in your privacy settings.",
                        ephemeral=True,
                    )
            else:
                await interaction.followup.send(safe_response)

        except Exception:
            log.exception(f"[{user_tag}] CRASH in _handle_chat")
            try:
                await interaction.followup.send(
                    "Something broke internally. Please try again.", ephemeral=True
                )
            except Exception:
                pass  # If the followup itself fails, there is nothing more to do.
