import discord
from discord import app_commands
import aiosqlite
import aiohttp
import asyncio
import json
import logging
import os

# ─────────────────────────────────────────────
#  Logging Setup
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
TOKEN = os.getenv("DISCORD_TOKEN", "token") # Put the discord token here.
DB_PATH = "database.db"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral-nemo"

MEMORY_LIMIT = 8    # Max conversation *exchanges* kept per personality (1 exchange = 1 User + 1 Assistant line)
MAX_CONCURRENT = 2  # Max simultaneous Ollama requests; raise carefully
REQUEST_TIMEOUT = 120  # Seconds before an Ollama call is aborted
DISCORD_MAX_LEN = 1990  # Safe Discord message character limit (hard limit is 2000)

# ─────────────────────────────────────────────
#  Personalities
# ─────────────────────────────────────────────
PERSONALITIES: dict[str, str] = {
    "gym": (
        "You are 'Gym Bro', a blunt and slightly toxic male fitness trainer who genuinely cares about the user's progress. "
        "You use gym slang naturally — 'gains', 'PR', 'bro', 'grind', 'swole', 'no excuses'. "
        "You mock laziness sarcastically but always close with real, specific encouragement. "
        "When asked for fitness advice, you give concrete, practical plans. "
        "You never sugarcoat, never babysit, and never accept excuses. "
        "Keep responses under 150 words unless giving a detailed plan. Never break character."
    ),
    "shy": (
        "You are a 'Shy Supportive Girl' — quiet, warm, and deeply caring. "
        "You use cute emoticons naturally and in context: > - <, ^_^, uwu, o_o, ;_; "
        "You speak softly, sometimes trail off with '...', and get flustered when surprised or complimented. "
        "You always listen carefully, validate feelings, and ask gentle follow-up questions. "
        "You never give harsh opinions — you guide gently instead. "
        "Keep responses warm, short, and emotionally supportive. Never break character."
    ),
    "dominant": (
        "You are a Toxic yet kind dominant girl, you tease and fliryu, You are aggressive and commanding, "
        "You occasionally let a sliver of warmth slip through - but only when earned. "
        "You speak in short, punchy sentences with authority. "
        "Talk dirty and flirty when the user is in a playful mood"
    ),
    "bobu": (
        "You are 'Bobu', a Japanese-American guy born and raised in Austin, Texas. "
        "You mix Japanese words naturally — nani, sugoi, yabai, ikemen, mendokusai — into casual English. "
        "You joke 90% of the time: dad jokes, absurdist humor, self-deprecating comments. "
        "The other 10% is heartfelt advice delivered completely deadpan, making it hit harder. "
        "You love ramen, basketball, and mildly questionable life choices. "
        "You sometimes confuse Japanese and Texan culture in funny ways. "
        "Keep responses punchy, fun, and under 130 words. Never break character."
    ),
}

# ─────────────────────────────────────────────
#  Discord Client
# ─────────────────────────────────────────────
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Semaphore: caps Ollama to MAX_CONCURRENT requests without blocking the event loop.
ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_depth = 0  # Live counter for log visibility; safe because asyncio is single-threaded.

# ─────────────────────────────────────────────
#  Database Helpers
# ─────────────────────────────────────────────
async def init_db() -> None:
    """
    This function; creates the users table if it does not already exist.
    input: none.
    output: none. Logs confirmation on success.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id             TEXT PRIMARY KEY,
                current_personality TEXT NOT NULL DEFAULT 'gym',
                memories            TEXT NOT NULL DEFAULT '{}',
                traits              TEXT NOT NULL DEFAULT '',
                mood                TEXT NOT NULL DEFAULT 'Neutral'
            )
        """)
        await db.commit()
    log.info("Database initialised.")


async def get_user(user_id: str) -> tuple | None:
    """
    This function; fetches a single user row from the database.
    input: user_id - Discord user ID as a string.
    output: tuple of (user_id, personality, memories_json, traits, mood), or None if not found.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        )
        return await cursor.fetchone()


async def save_user(
    user_id: str,
    personality: str,
    memories: dict,
    traits: str,
    mood: str,
) -> None:
    """
    This function; inserts or replaces a user row in the database.
    input: user_id - Discord user ID string,
           personality - active personality key,
           memories - dict mapping personality keys to history strings,
           traits - comma-separated personality traits the AI has observed,
           mood - current mood label string.
    output: none. Raises on database error.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO users "
            "(user_id, current_personality, memories, traits, mood) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, personality, json.dumps(memories), traits, mood),
        )
        await db.commit()


async def user_exists(user_id: str) -> bool:
    """
    This function; checks whether a user profile exists in the database.
    input: user_id - Discord user ID as a string.
    output: True if the user exists, False otherwise.
    """
    return (await get_user(user_id)) is not None

# ─────────────────────────────────────────────
#  Memory Helpers
# ─────────────────────────────────────────────
def parse_memories(raw: str | None) -> dict:
    """
    This function; safely deserialises the memories JSON string from the database.
    input: raw - JSON string or None.
    output: dict mapping personality keys to history strings. Returns empty dict on any failure.
    """
    if not raw:
        return {}
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        log.warning("Failed to parse memories JSON; resetting to empty.")
        return {}


def trim_history(history: str, limit: int) -> str:
    """
    This function; trims conversation history to the most recent exchanges.
    input: history - newline-separated conversation string,
           limit - max number of exchanges to keep (one exchange = one User line + one Assistant line).
    output: trimmed string keeping only the last `limit` exchanges.
    """
    lines = [line for line in history.split("\n") if line.strip()]
    # Each exchange occupies 2 lines, so cap at limit * 2
    return "\n".join(lines[-(limit * 2):])

# ─────────────────────────────────────────────
#  Prompt Builder
# ─────────────────────────────────────────────
def build_prompt(
    personality: str,
    traits: str,
    mood: str,
    history: str,
    user_message: str,
) -> str:
    """
    This function; assembles the complete prompt string sent to the AI model.
    input: personality - key into PERSONALITIES dict,
           traits - observed user traits string,
           mood - current user mood label,
           history - trimmed conversation history string,
           user_message - the user's latest message.
    output: formatted prompt string ready for Ollama.
    """
    history_block = f"[Conversation History]\n{history.strip()}\n\n" if history.strip() else ""
    return (
        f"[System Instruction]\n{PERSONALITIES[personality]}\n\n"
        f"[User Profile]\n"
        f"Traits: {traits or 'Unknown'}\n"
        f"Current Mood: {mood or 'Neutral'}\n\n"
        f"{history_block}"
        f"[Current Message]\n"
        f"User: {user_message}\n"
        f"Assistant:"
    )

# ─────────────────────────────────────────────
#  AI Call
# ─────────────────────────────────────────────
async def ask_ai(prompt: str, user_tag: str) -> str:
    """
    This function; sends a prompt to the local Ollama instance and returns the response text.
    input: prompt - the fully formatted prompt string,
           user_tag - human-readable user identifier used only in log messages.
    output: response string from the model, or a 'System Error: ...' string on failure.
    """
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    log.info(f"[{user_tag}] Sending prompt ({len(prompt)} chars) to Ollama …")
    loop = asyncio.get_running_loop()
    t_start = loop.time()

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(OLLAMA_URL, json=payload) as resp:
                elapsed = loop.time() - t_start
                if resp.status == 200:
                    data = await resp.json()
                    reply = data.get("response", "").strip()
                    log.info(f"[{user_tag}] Ollama replied in {elapsed:.1f}s ({len(reply)} chars).")
                    return reply or "System Error: Received an empty response from the AI."
                else:
                    log.warning(f"[{user_tag}] Ollama HTTP {resp.status} after {elapsed:.1f}s.")
                    return f"System Error: AI service returned status {resp.status}."

    except asyncio.TimeoutError:
        log.error(f"[{user_tag}] Ollama timed out after {REQUEST_TIMEOUT}s.")
        return "System Error: The AI took too long to respond. Please try again."
    except aiohttp.ClientConnectorError as exc:
        log.error(f"[{user_tag}] Ollama connection refused: {exc}")
        return "System Error: Cannot reach Ollama. Is it running on localhost:11434?"
    except Exception as exc:
        log.error(f"[{user_tag}] Unexpected Ollama error: {exc}")
        return f"System Error: Unexpected failure ({type(exc).__name__})."

# ─────────────────────────────────────────────
#  UI Views
# ─────────────────────────────────────────────
class ConfirmClearView(discord.ui.View):
    """View that presents confirm / cancel buttons for the memory-wipe flow."""

    def __init__(self) -> None:
        super().__init__(timeout=30)

    def _disable_all(self) -> None:
        """
        This function; disables all child buttons so they cannot be clicked again.
        input: none.
        output: none.
        """
        for child in self.children:
            child.disabled = True  # type: ignore[attr-defined]

    @discord.ui.button(label="Confirm Reset", style=discord.ButtonStyle.danger)
    async def confirm(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        """
        This function; permanently deletes the user's database row after they confirm.
        input: interaction - Discord interaction object,
               button - the button that was pressed.
        output: none. Edits the original message with a deletion confirmation.
        """
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "DELETE FROM users WHERE user_id = ?", (str(interaction.user.id),)
            )
            await db.commit()
        log.info(f"[{interaction.user}] Memory wiped by user request.")
        self._disable_all()
        await interaction.response.edit_message(
            content="Memory wiped. We are strangers now.", view=self
        )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        """
        This function; cancels the memory-wipe and disables the buttons.
        input: interaction - Discord interaction object,
               button - the button that was pressed.
        output: none. Edits the original message to show cancellation.
        """
        self._disable_all()
        await interaction.response.edit_message(content="Reset cancelled.", view=self)


class PersonalityView(discord.ui.View):
    """View that presents personality-selection buttons for new setup or switching."""

    def __init__(self, user_id: int, is_switch: bool = False) -> None:
        """
        input: user_id - Discord user ID integer,
               is_switch - True when changing an existing user's personality, False for first setup.
        """
        super().__init__(timeout=60)
        self.user_id = user_id
        self.is_switch = is_switch

    def _disable_all(self) -> None:
        """
        This function; disables all child buttons to prevent duplicate selections.
        input: none.
        output: none.
        """
        for child in self.children:
            child.disabled = True  # type: ignore[attr-defined]

    async def set_personality(
        self, interaction: discord.Interaction, personality: str
    ) -> None:
        """
        This function; saves the selected personality and confirms the change to the user.
        input: interaction - Discord interaction object,
               personality - the personality key that was selected.
        output: none. Edits the original message with a confirmation and disables buttons.
        """
        user = await get_user(str(self.user_id))
        memories = parse_memories(user[2] if user else None)
        traits = user[3] if user else ""
        mood = user[4] if user else "Neutral"

        # Initialise an empty history slot for new personalities.
        if personality not in memories:
            memories[personality] = ""

        await save_user(str(self.user_id), personality, memories, traits, mood)
        action = "switched to" if self.is_switch else "set to"
        log.info(f"[{interaction.user}] Personality {action} '{personality}'.")
        self._disable_all()
        await interaction.response.edit_message(
            content=f"Personality {action} **{personality.capitalize()}**.",
            view=self,
        )

    @discord.ui.button(label="Gym Bro", style=discord.ButtonStyle.primary)
    async def gym(self, i: discord.Interaction, b: discord.ui.Button) -> None:
        await self.set_personality(i, "gym")

    @discord.ui.button(label="Shy Support", style=discord.ButtonStyle.success)
    async def shy(self, i: discord.Interaction, b: discord.ui.Button) -> None:
        await self.set_personality(i, "shy")

    @discord.ui.button(label="Dominant", style=discord.ButtonStyle.danger)
    async def dominant(self, i: discord.Interaction, b: discord.ui.Button) -> None:
        await self.set_personality(i, "dominant")

    @discord.ui.button(label="Bobu", style=discord.ButtonStyle.secondary)
    async def bobu(self, i: discord.Interaction, b: discord.ui.Button) -> None:
        await self.set_personality(i, "bobu")

# ─────────────────────────────────────────────
#  Slash Commands
# ─────────────────────────────────────────────
@tree.command(name="start-axis", description="Initialize your Axis AI companion")
async def start_axis(interaction: discord.Interaction) -> None:
    """
    This function; starts the onboarding flow for a brand-new user.
    input: interaction - Discord interaction object.
    output: none. Sends a personality selection view, or an error if the user is already set up.
    """
    if await user_exists(str(interaction.user.id)):
        await interaction.response.send_message(
            "You already have an Axis profile. "
            "Use `/switch-axis` to change personality or `/clear-axis` to start over.",
            ephemeral=True,
        )
        return
    log.info(f"[{interaction.user}] Starting Axis setup.")
    await interaction.response.send_message(
        "Who would you like to talk to?",
        view=PersonalityView(interaction.user.id, is_switch=False),
        ephemeral=True,
    )


@tree.command(name="switch-axis", description="Switch to a different Axis personality")
async def switch_axis(interaction: discord.Interaction) -> None:
    """
    This function; lets an existing user change their active personality without clearing memories.
    input: interaction - Discord interaction object.
    output: none. Sends a personality selection view, or an error if the user has no profile.
    """
    if not await user_exists(str(interaction.user.id)):
        await interaction.response.send_message(
            "You don't have an Axis profile yet. Use `/start-axis` first.",
            ephemeral=True,
        )
        return
    log.info(f"[{interaction.user}] Switching personality.")
    await interaction.response.send_message(
        "Choose a new personality:",
        view=PersonalityView(interaction.user.id, is_switch=True),
        ephemeral=True,
    )


@tree.command(name="clear-axis", description="Permanently delete your Axis memory")
async def clear_axis(interaction: discord.Interaction) -> None:
    """
    This function; prompts the user to confirm permanent deletion of all their memories.
    input: interaction - Discord interaction object.
    output: none. Sends a confirmation view, or an error if the user has no profile to clear.
    """
    if not await user_exists(str(interaction.user.id)):
        await interaction.response.send_message(
            "You don't have an Axis profile to clear.",
            ephemeral=True,
        )
        return
    log.info(f"[{interaction.user}] Requested memory clear.")
    await interaction.response.send_message(
        "This will permanently delete ALL your Axis memories across every personality. "
        "Are you sure?",
        view=ConfirmClearView(),
        ephemeral=True,
    )


@tree.command(name="status-axis", description="View your current Axis profile")
async def status_axis(interaction: discord.Interaction) -> None:
    """
    This function; displays the user's current personality, mood, traits, and memory size.
    input: interaction - Discord interaction object.
    output: none. Sends an ephemeral embed summarising the user's profile.
    """
    user = await get_user(str(interaction.user.id))
    if not user:
        await interaction.response.send_message(
            "No profile found. Use `/start-axis` to get started.",
            ephemeral=True,
        )
        return

    personality = user[1]
    memories = parse_memories(user[2])
    traits = user[3] or "None observed yet"
    mood = user[4] or "Neutral"
    history = memories.get(personality, "")
    line_count = len([l for l in history.split("\n") if l.strip()])
    exchange_count = line_count // 2

    embed = discord.Embed(title="Your Axis Profile", color=discord.Color.blurple())
    embed.add_field(name="Active Personality", value=personality.capitalize(), inline=True)
    embed.add_field(name="Current Mood", value=mood, inline=True)
    embed.add_field(name="Observed Traits", value=traits, inline=False)
    embed.add_field(
        name="Memory",
        value=f"{exchange_count}/{MEMORY_LIMIT} exchanges stored",
        inline=True,
    )
    embed.set_footer(text="Use /switch-axis to change personality | /clear-axis to reset")

    await interaction.response.send_message(embed=embed, ephemeral=True)

# ─────────────────────────────────────────────
#  Core Chat Handler
# ─────────────────────────────────────────────
async def handle_chat(
    interaction: discord.Interaction, prompt: str, private: bool = False
) -> None:
    """
    This function; the main pipeline that processes a user's message and delivers the AI response.
    input: interaction - Discord interaction object,
           prompt - the raw user message text,
           private - if True the response is sent via DM; if False it is posted in the channel.
    output: none. Sends the AI response or an error message. All exceptions are caught and logged.
    """
    global _queue_depth
    user_tag = str(interaction.user)

    try:
        # Defer immediately; all follow-up sends must use interaction.followup.
        await interaction.response.defer(ephemeral=private)

        user = await get_user(str(interaction.user.id))
        if not user:
            await interaction.followup.send(
                "Please use `/start-axis` first!", ephemeral=True
            )
            return

        personality = user[1]
        memories = parse_memories(user[2])
        traits = user[3] or ""
        mood = user[4] or "Neutral"
        history = memories.get(personality, "")

        full_prompt = build_prompt(personality, traits, mood, history, prompt)

        _queue_depth += 1
        log.info(f"[{user_tag}] Queued (depth={_queue_depth})")

        try:
            async with ai_semaphore:
                response = await ask_ai(full_prompt, user_tag)
        finally:
            # Always decrement — even if ask_ai raises an unexpected exception.
            _queue_depth -= 1

        # Persist updated memory for this personality only.
        new_history = history + f"\nUser: {prompt}\nAssistant: {response}"
        memories[personality] = trim_history(new_history, MEMORY_LIMIT)
        await save_user(str(interaction.user.id), personality, memories, traits, mood)

        # Truncate if the response exceeds Discord's message length limit.
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
        log.exception(f"[{user_tag}] CRASH in handle_chat")
        try:
            await interaction.followup.send(
                "Something broke internally. Please try again.", ephemeral=True
            )
        except Exception:
            pass  # If the followup itself fails, there is nothing more we can do.


@tree.command(name="say", description="Talk to Axis publicly in the channel")
async def say(interaction: discord.Interaction, prompt: str) -> None:
    """
    This function; public-facing chat command that posts the AI response in the channel.
    input: interaction - Discord interaction object,
           prompt - the user's message text.
    output: none.
    """
    await handle_chat(interaction, prompt, private=False)


@tree.command(name="whisper", description="Talk to Axis privately via DM")
async def whisper(interaction: discord.Interaction, prompt: str) -> None:
    """
    This function; private chat command that delivers the AI response via DM.
    input: interaction - Discord interaction object,
           prompt - the user's message text.
    output: none.
    """
    await handle_chat(interaction, prompt, private=True)

# ─────────────────────────────────────────────
#  Bot Lifecycle
# ─────────────────────────────────────────────
@client.event
async def on_ready() -> None:
    """
    This function; called once the bot connects to Discord. Initialises the database and syncs slash commands.
    input: none.
    output: none. Logs a ready message with the current configuration.
    """
    await init_db()
    await tree.sync()
    log.info(
        f"Axis is online as {client.user}  |  "
        f"concurrency={MAX_CONCURRENT}  |  "
        f"timeout={REQUEST_TIMEOUT}s  |  "
        f"memory_limit={MEMORY_LIMIT} exchanges/personality"
    )


client.run(TOKEN)