import logging

import discord

from handlers.database import Database
from personalities import PERSONALITIES, PERSONALITY_NAMES, PERSONALITY_DESCRIPTIONS
from ui.modals import NameModal

log = logging.getLogger("Axis")


# ─── Step 3: Name ─────────────────────────────────────────────────────────────

class SetNameView(discord.ui.View):
    """Step 3 of /start-axis: offer to set a name or skip."""

    def __init__(self, user_id: int, db: Database) -> None:
        super().__init__(timeout=120)
        self.user_id = user_id
        self.db      = db

    def _disable_all(self) -> None:
        for child in self.children:
            child.disabled = True  # type: ignore[attr-defined]

    @discord.ui.button(label="Set My Name", style=discord.ButtonStyle.primary)
    async def set_name(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await interaction.response.send_modal(
            NameModal(self.user_id, self.db, from_setup=True)
        )

    @discord.ui.button(label="Skip for Now", style=discord.ButtonStyle.secondary)
    async def skip(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        user      = await self.db.get_user(str(self.user_id))
        companion = PERSONALITY_NAMES.get(
            self.db.row_get(user, 1, "gym") if user else "gym", "your companion"
        )
        self._disable_all()
        await interaction.response.edit_message(
            content=(
                f"All set! **{companion}** is ready.\n"
                f"Use `/say` to start chatting, or `/setname` to tell them your name later. 🎉"
            ),
            view=self,
        )


# ─── Step 2: Gender ───────────────────────────────────────────────────────────

class GenderSelect(discord.ui.Select):
    """Dropdown for gender selection — step 2 of setup or standalone /set-gender."""

    def __init__(self, user_id: int, db: Database, from_setup: bool = False) -> None:
        self.user_id    = user_id
        self.db         = db
        self.from_setup = from_setup
        options = [
            discord.SelectOption(label="Male",   value="male",   emoji="♂️"),
            discord.SelectOption(label="Female", value="female", emoji="♀️"),
        ]
        super().__init__(
            placeholder="Select your gender …",
            options=options,
            min_values=1,
            max_values=1,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        gender = self.values[0]
        user   = await self.db.get_user(str(self.user_id))
        if not user:
            await interaction.response.edit_message(
                content="Something went wrong — please run `/start-axis` again.", view=None
            )
            return

        await self.db.save_user(
            str(self.user_id),
            self.db.row_get(user, 1, "gym"),
            self.db.parse_memories(user[2] if len(user) > 2 else None),
            self.db.row_get(user, 3, ""),
            self.db.row_get(user, 4, "Neutral"),
            self.db.row_get(user, 5, ""),
            int(user[6]) if len(user) > 6 and user[6] else 0,
            gender,                                                         # ← updated field
        )
        log.info(f"[{interaction.user}] Gender set to '{gender}'.")

        if self.from_setup:
            await interaction.response.edit_message(
                content=(
                    "Almost done! Do you want to tell your companion your name? "
                    "They'll remember it forever."
                ),
                view=SetNameView(self.user_id, self.db),
            )
        else:
            await interaction.response.edit_message(
                content=(
                    f"Gender updated to **{gender.capitalize()}**. "
                    f"Your companion will address you correctly from now on."
                ),
                view=None,
            )


class GenderView(discord.ui.View):
    """Wraps GenderSelect for step 2 of setup or standalone /set-gender."""

    def __init__(self, user_id: int, db: Database, from_setup: bool = False) -> None:
        super().__init__(timeout=60)
        self.add_item(GenderSelect(user_id, db, from_setup=from_setup))


# ─── Step 1: Personality ──────────────────────────────────────────────────────

class PersonalitySelect(discord.ui.Select):
    """Step 1 dropdown — choose a companion personality."""

    def __init__(self, user_id: int, db: Database) -> None:
        self.user_id = user_id
        self.db      = db
        options = [
            discord.SelectOption(
                label=f"{PERSONALITY_NAMES[key]} — {key.capitalize()}",
                value=key,
                description=PERSONALITY_DESCRIPTIONS[key],
            )
            for key in PERSONALITIES
        ]
        super().__init__(
            placeholder="Choose your companion …",
            options=options,
            min_values=1,
            max_values=1,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        personality = self.values[0]
        memories: dict = {personality: ""}

        # Create the row with defaults; subsequent steps fill gender and name.
        await self.db.save_user(
            str(self.user_id), personality, memories, "", "Neutral", "", 0, "unspecified"
        )
        log.info(f"[{interaction.user}] Personality set to '{personality}' during setup.")

        name = PERSONALITY_NAMES[personality]
        await interaction.response.edit_message(
            content=(
                f"Great — **{name}** is your companion!\n\n"
                f"Step 2: What's your gender? This helps them address you correctly."
            ),
            view=GenderView(self.user_id, self.db, from_setup=True),
        )


class PersonalityView(discord.ui.View):
    """Wraps PersonalitySelect for step 1 of the /start-axis flow."""

    def __init__(self, user_id: int, db: Database) -> None:
        super().__init__(timeout=60)
        self.add_item(PersonalitySelect(user_id, db))


# ─── Memory wipe confirmation ─────────────────────────────────────────────────

class ConfirmClearView(discord.ui.View):
    """Confirm / cancel buttons for the /clear-axis memory-wipe flow."""

    def __init__(self, db: Database) -> None:
        super().__init__(timeout=30)
        self.db = db

    def _disable_all(self) -> None:
        for child in self.children:
            child.disabled = True  # type: ignore[attr-defined]

    @discord.ui.button(label="Confirm Reset", style=discord.ButtonStyle.danger)
    async def confirm(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        await self.db.delete_user(str(interaction.user.id))
        log.info(f"[{interaction.user}] Memory wiped by user request.")
        self._disable_all()
        await interaction.response.edit_message(
            content="Memory wiped. We are strangers now.", view=self
        )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        self._disable_all()
        await interaction.response.edit_message(content="Reset cancelled.", view=self)
