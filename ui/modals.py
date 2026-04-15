import logging

import discord

from handlers.database import Database
from personalities import PERSONALITY_NAMES

log = logging.getLogger("Axis")


class NameModal(discord.ui.Modal, title="What's your name?"):
    """Text-input modal used in /start-axis step 3 and standalone /setname."""

    name_input = discord.ui.TextInput(
        label="Your name",
        placeholder="Enter the name you want to be called …",
        min_length=1,
        max_length=50,
    )

    def __init__(self, user_id: int, db: Database, from_setup: bool = False) -> None:
        super().__init__()
        self.user_id    = user_id
        self.db         = db
        self.from_setup = from_setup

    async def on_submit(self, interaction: discord.Interaction) -> None:
        name = self.name_input.value.strip()
        uid  = str(self.user_id)
        user = await self.db.get_user(uid)

        if not user:
            await interaction.response.send_message(
                "Profile not found. Please run `/start-axis` again.", ephemeral=True
            )
            return

        await self.db.save_user(
            uid,
            self.db.row_get(user, 1, "gym"),
            self.db.parse_memories(user[2] if len(user) > 2 else None),
            self.db.row_get(user, 3, ""),
            self.db.row_get(user, 4, "Neutral"),
            name,                                                          # ← updated field
            int(user[6]) if len(user) > 6 and user[6] else 0,
            self.db.row_gender(user),
        )
        log.info(f"[{interaction.user}] Name saved as '{name}' (from_setup={self.from_setup}).")

        if self.from_setup:
            companion = PERSONALITY_NAMES.get(self.db.row_get(user, 1, "gym"), "your companion")
            await interaction.response.send_message(
                f"You're all set! **{companion}** will always call you **{name}**.\n"
                f"Use `/say` to start chatting. 🎉",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                f"Got it — I'll always remember your name is **{name}**.", ephemeral=True
            )
