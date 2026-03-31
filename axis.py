import discord
from discord import app_commands
import aiosqlite
import aiohttp
import asyncio
import json
import logging
import os
import yt_dlp
from collections import defaultdict

# =============================================
#  GENERAL
# =============================================
# ---------------------------------------------
#  Logging
# ---------------------------------------------
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

# ---------------------------------------------
#  Config
# ---------------------------------------------
TOKEN = os.getenv("DISCORD_TOKEN", "token") #Put your Discord bot token here.
DB_PATH = "database.db"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral-nemo"

MEMORY_LIMIT = 12 #Max conversation exchanges kept per personality per user.
MAX_CONCURRENT = 2 #Max simultaneous Ollama requests. Raise with caution.
REQUEST_TIMEOUT = 120 #Seconds before an Ollama call times out.
DISCORD_MAX_LEN = 1990 #Character cap per Discord message (hard limit is 2000).

# ---------------------------------------------
#  Discord Client
# ---------------------------------------------
intents = discord.Intents.default()
intents.voice_states = True #Required to detect which voice channel the user is in.
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT) #Limits concurrent Ollama calls without blocking the event loop.
_queue_depth = 0 #Tracks how many AI requests are currently waiting or running.

# =============================================
#  AI / CHATBOT
# =============================================
# ---------------------------------------------
#  Personalities
# ---------------------------------------------
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
        "You are a Toxic yet kind dominant girl, you tease and flirt. You are aggressive and commanding. "
        "You occasionally let a sliver of warmth slip through - but only when earned. "
        "You speak in short, punchy sentences with authority. "
        "Talk dirty and flirty when the user is in a playful mood."
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

# ---------------------------------------------
#  Database Helpers
# ---------------------------------------------
async def init_db() -> None:
    """
    This function; creates the users table and migrates any missing columns for existing databases.
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
                mood                TEXT NOT NULL DEFAULT 'Neutral',
                user_name           TEXT NOT NULL DEFAULT '',
                prompt_count        INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Migrate existing tables that are missing the newer columns.
        for col, definition in [
            ("user_name", "TEXT NOT NULL DEFAULT ''"),
            ("prompt_count", "INTEGER NOT NULL DEFAULT 0"),
        ]:
            try:
                await db.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
            except Exception:
                pass #Column already exists; safe to ignore.
        await db.commit()
    log.info("Database initialised.")


async def get_user(user_id: str) -> tuple | None:
    """
    This function; fetches a single user row from the database.
    input: user_id - Discord user ID as a string.
    output: tuple of (user_id, personality, memories_json, traits, mood, user_name, prompt_count), or None if not found.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        return await cursor.fetchone()


async def save_user(
    user_id: str,
    personality: str,
    memories: dict,
    traits: str,
    mood: str,
    user_name: str = "",
    prompt_count: int = 0,
) -> None:
    """
    This function; inserts or replaces a user row in the database.
    input: user_id - Discord user ID string,
           personality - active personality key,
           memories - dict mapping personality keys to history strings,
           traits - observed user traits string,
           mood - current mood label string,
           user_name - the user's remembered name,
           prompt_count - total number of AI prompts sent by this user.
    output: none. Raises on database error.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO users "
            "(user_id, current_personality, memories, traits, mood, user_name, prompt_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, personality, json.dumps(memories), traits, mood, user_name, prompt_count),
        )
        await db.commit()


async def user_exists(user_id: str) -> bool:
    """
    This function; checks whether a user profile exists in the database.
    input: user_id - Discord user ID as a string.
    output: True if the user exists, False otherwise.
    """
    return (await get_user(user_id)) is not None

# ---------------------------------------------
#  Memory Helpers
# ---------------------------------------------
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
    return "\n".join(lines[-(limit * 2):]) #Each exchange = 2 lines.

# ---------------------------------------------
#  Prompt Builder
# ---------------------------------------------
def build_prompt(
    personality: str,
    traits: str,
    mood: str,
    history: str,
    user_message: str,
    user_name: str = "",
) -> str:
    """
    This function; assembles the complete prompt string sent to the AI model.
    input: personality - key into PERSONALITIES dict,
           traits - observed user traits string,
           mood - current user mood label,
           history - trimmed conversation history string,
           user_message - the user's latest message,
           user_name - the user's remembered name (empty string if unknown).
    output: formatted prompt string ready for Ollama.
    """
    history_block = f"[Conversation History]\n{history.strip()}\n\n" if history.strip() else ""
    name_line = (
        f"User's Name: {user_name} — always address them by this name and never forget it.\n"
        if user_name
        else "User's Name: Unknown — learn it naturally if they share it.\n"
    )
    return (
        f"[System Instruction]\n{PERSONALITIES[personality]}\n\n"
        f"[User Profile]\n"
        f"{name_line}"
        f"Traits: {traits or 'Unknown'}\n"
        f"Current Mood: {mood or 'Neutral'}\n\n"
        f"{history_block}"
        f"[Current Message]\n"
        f"User: {user_message}\n"
        f"Assistant:"
    )

# ---------------------------------------------
#  AI Call
# ---------------------------------------------
async def ask_ai(prompt: str, user_tag: str) -> str:
    """
    This function; sends a prompt to the local Ollama instance and returns the response text.
    input: prompt - the fully formatted prompt string,
           user_tag - human-readable user identifier used only in log messages.
    output: response string from the model, or a 'System Error: ...' string on failure.
    """
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    log.info(f"[{user_tag}] Sending prompt ({len(prompt)} chars) to Ollama ...")
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

# ---------------------------------------------
#  UI Views
# ---------------------------------------------
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
            await db.execute("DELETE FROM users WHERE user_id = ?", (str(interaction.user.id),))
            await db.commit()
        log.info(f"[{interaction.user}] Memory wiped by user request.")
        self._disable_all()
        await interaction.response.edit_message(content="Memory wiped. We are strangers now.", view=self)

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

    async def set_personality(self, interaction: discord.Interaction, personality: str) -> None:
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
        user_name = user[5] if user else ""
        prompt_count = user[6] if user else 0

        if personality not in memories:
            memories[personality] = ""

        await save_user(str(self.user_id), personality, memories, traits, mood, user_name, prompt_count)
        action = "switched to" if self.is_switch else "set to"
        log.info(f"[{interaction.user}] Personality {action} '{personality}'.")
        self._disable_all()
        await interaction.response.edit_message(
            content=f"Personality {action} **{personality.capitalize()}**.", view=self
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

# ---------------------------------------------
#  Slash Commands (AI)
# ---------------------------------------------
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
            "You don't have an Axis profile yet. Use `/start-axis` first.", ephemeral=True
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
            "You don't have an Axis profile to clear.", ephemeral=True
        )
        return
    log.info(f"[{interaction.user}] Requested memory clear.")
    await interaction.response.send_message(
        "This will permanently delete ALL your Axis memories across every personality. Are you sure?",
        view=ConfirmClearView(),
        ephemeral=True,
    )


@tree.command(name="status-axis", description="View your current Axis profile")
async def status_axis(interaction: discord.Interaction) -> None:
    """
    This function; displays the user's current personality, name, mood, traits, and memory size.
    input: interaction - Discord interaction object.
    output: none. Sends an ephemeral embed summarising the user's profile.
    """
    user = await get_user(str(interaction.user.id))
    if not user:
        await interaction.response.send_message(
            "No profile found. Use `/start-axis` to get started.", ephemeral=True
        )
        return

    personality = user[1]
    memories = parse_memories(user[2])
    traits = user[3] or "None observed yet"
    mood = user[4] or "Neutral"
    user_name = user[5] or "Not set — use /setname"
    prompt_count = user[6] or 0
    history = memories.get(personality, "")
    exchange_count = len([l for l in history.split("\n") if l.strip()]) // 2

    embed = discord.Embed(title="Your Axis Profile", color=discord.Color.blurple())
    embed.add_field(name="Active Personality", value=personality.capitalize(), inline=True)
    embed.add_field(name="Current Mood", value=mood, inline=True)
    embed.add_field(name="Your Name", value=user_name, inline=True)
    embed.add_field(name="Observed Traits", value=traits, inline=False)
    embed.add_field(name="Memory", value=f"{exchange_count}/{MEMORY_LIMIT} exchanges stored", inline=True)
    embed.add_field(name="Total Prompts", value=str(prompt_count), inline=True)
    embed.set_footer(text="/switch-axis to change personality | /clear-axis to reset | /setname to set your name")
    await interaction.response.send_message(embed=embed, ephemeral=True)


@tree.command(name="setname", description="Tell Axis your name so it never forgets")
async def setname(interaction: discord.Interaction, name: str) -> None:
    """
    This function; saves the user's name persistently so every AI personality remembers it.
    input: interaction - Discord interaction object,
           name - the name the user wants to be addressed by.
    output: none. Confirms the name was saved, or prompts setup if no profile exists.
    """
    user = await get_user(str(interaction.user.id))
    if not user:
        await interaction.response.send_message("Please use `/start-axis` first!", ephemeral=True)
        return
    memories = parse_memories(user[2])
    await save_user(str(interaction.user.id), user[1], memories, user[3], user[4], name.strip(), user[6] or 0)
    log.info(f"[{interaction.user}] Name set to '{name.strip()}'.")
    await interaction.response.send_message(
        f"Got it — I'll always remember your name is **{name.strip()}**.", ephemeral=True
    )


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

# ---------------------------------------------
#  Core Chat Handler
# ---------------------------------------------
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
        await interaction.response.defer(ephemeral=private)

        user = await get_user(str(interaction.user.id))
        if not user:
            await interaction.followup.send("Please use `/start-axis` first!", ephemeral=True)
            return

        personality = user[1]
        memories = parse_memories(user[2])
        traits = user[3] or ""
        mood = user[4] or "Neutral"
        user_name = user[5] or ""
        prompt_count = (user[6] or 0) + 1 #Increment on each successful chat.
        history = memories.get(personality, "")

        full_prompt = build_prompt(personality, traits, mood, history, prompt, user_name)

        _queue_depth += 1
        log.info(f"[{user_tag}] Queued (depth={_queue_depth})")

        try:
            async with ai_semaphore:
                response = await ask_ai(full_prompt, user_tag)
        finally:
            _queue_depth -= 1 #Always decrement, even on exception.

        new_history = history + f"\nUser: {prompt}\nAssistant: {response}"
        memories[personality] = trim_history(new_history, MEMORY_LIMIT)
        await save_user(str(interaction.user.id), personality, memories, traits, mood, user_name, prompt_count)

        safe_response = (
            response[:DISCORD_MAX_LEN] + "…" if len(response) > DISCORD_MAX_LEN else response
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
            await interaction.followup.send("Something broke internally. Please try again.", ephemeral=True)
        except Exception:
            pass #If the followup itself fails there is nothing more we can do.

# =============================================
#  MUSIC
# =============================================
# ---------------------------------------------
#  Queue & State
# ---------------------------------------------
music_queues: dict[int, list[dict]] = defaultdict(list) #Per-guild song queue keyed by guild_id.
voice_clients: dict[int, discord.VoiceClient] = {} #Active voice clients keyed by guild_id.

YDL_OPTIONS = {
    "format": "bestaudio/best",
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "source_address": "0.0.0.0",
}

FFMPEG_OPTIONS = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options": "-vn",
}

# ---------------------------------------------
#  Music Helpers
# ---------------------------------------------
async def fetch_track(query: str, requester: str) -> dict | None:
    """
    This function; searches YouTube or fetches a direct URL and returns track metadata.
    input: query - a YouTube search term or full URL,
           requester - display name of the user who requested the track.
    output: dict with keys title, webpage_url, duration, requester — or None on failure.
    """
    loop = asyncio.get_running_loop()
    try:
        with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
            if query.startswith("http://") or query.startswith("https://"):
                info = await loop.run_in_executor(None, lambda: ydl.extract_info(query, download=False))
            else:
                info = await loop.run_in_executor(
                    None, lambda: ydl.extract_info(f"ytsearch1:{query}", download=False)
                )
                entries = info.get("entries", [])
                if not entries:
                    return None
                info = entries[0]
        return {
            "title": info.get("title", "Unknown"),
            "webpage_url": info.get("webpage_url", query),
            "duration": info.get("duration", 0),
            "requester": requester,
        }
    except Exception as exc:
        log.error(f"yt_dlp fetch error for '{query}': {exc}")
        return None


async def get_stream_url(webpage_url: str) -> str | None:
    """
    This function; extracts a fresh audio stream URL from a YouTube page URL.
    input: webpage_url - canonical YouTube watch URL for the track.
    output: direct streamable audio URL string, or None on failure.
    """
    loop = asyncio.get_running_loop()
    try:
        with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
            info = await loop.run_in_executor(None, lambda: ydl.extract_info(webpage_url, download=False))
        return info.get("url")
    except Exception as exc:
        log.error(f"yt_dlp stream error for '{webpage_url}': {exc}")
        return None


async def play_next(guild_id: int) -> None:
    """
    This function; plays the next track in the guild queue, or does nothing if the queue is empty.
    input: guild_id - Discord guild ID whose queue should advance.
    output: none. Starts audio playback and schedules the next track via the after callback.
    """
    vc = voice_clients.get(guild_id)
    if not vc or not vc.is_connected():
        return
    if vc.is_playing(): #Don't interrupt a track that is already running.
        return

    queue = music_queues[guild_id]
    if not queue:
        return

    track = queue[0] #Item stays in queue until playback ends; popped inside after_playing.
    stream_url = await get_stream_url(track["webpage_url"])

    if not stream_url: #Skip unplayable tracks and try the next one.
        queue.pop(0)
        await play_next(guild_id)
        return

    source = discord.FFmpegPCMAudio(stream_url, **FFMPEG_OPTIONS)

    def after_playing(error: Exception | None) -> None:
        if error:
            log.error(f"[Guild {guild_id}] Playback error: {error}")
        if music_queues[guild_id]: #Guard against queue being cleared mid-play.
            music_queues[guild_id].pop(0)
        asyncio.run_coroutine_threadsafe(play_next(guild_id), client.loop)

    vc.play(source, after=after_playing)
    log.info(f"[Guild {guild_id}] Now playing: {track['title']}")


def format_duration(seconds: int) -> str:
    """
    This function; converts a duration in seconds to a MM:SS display string.
    input: seconds - total duration in seconds.
    output: formatted string in MM:SS format.
    """
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"

# ---------------------------------------------
#  Slash Commands (Music)
# ---------------------------------------------
@tree.command(name="play", description="Play a song in your voice channel (search or URL)")
async def play(interaction: discord.Interaction, query: str) -> None:
    """
    This function; searches or fetches a track and adds it to the guild queue.
    input: interaction - Discord interaction object,
           query - a search term (e.g. 'i quit pewdiepie') or a direct YouTube URL.
    output: none. Sends an embed confirming whether the track is now playing or queued.
    """
    if not interaction.guild:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    voice_state = interaction.user.voice  # type: ignore[union-attr]
    if not voice_state or not voice_state.channel:
        await interaction.response.send_message("You need to be in a voice channel first.", ephemeral=True)
        return

    await interaction.response.defer()

    guild_id = interaction.guild.id
    channel = voice_state.channel

    vc = voice_clients.get(guild_id)
    if not vc or not vc.is_connected():
        vc = await channel.connect()
        voice_clients[guild_id] = vc
    elif vc.channel != channel:
        await vc.move_to(channel)

    track = await fetch_track(query, str(interaction.user))
    if not track:
        await interaction.followup.send("Could not find anything for that query. Try a different search or URL.")
        return

    already_playing = vc.is_playing() or vc.is_paused()
    music_queues[guild_id].append(track)
    position = len(music_queues[guild_id])

    embed = discord.Embed(color=discord.Color.green())
    embed.description = f"**{track['title']}**"
    embed.add_field(name="Duration", value=format_duration(track["duration"]), inline=True)
    embed.add_field(name="Requested by", value=track["requester"], inline=True)

    if already_playing:
        embed.title = "Added to Queue 🎶"
        embed.add_field(name="Position", value=f"#{position}", inline=True)
    else:
        embed.title = "Now Playing 🎵"
        await play_next(guild_id)

    await interaction.followup.send(embed=embed)
    log.info(f"[{interaction.user}] Queued: {track['title']}")


@tree.command(name="play-queue", description="Show the current music queue")
async def play_queue(interaction: discord.Interaction) -> None:
    """
    This function; displays all tracks in the guild's current play queue.
    input: interaction - Discord interaction object.
    output: none. Sends an embed listing queued tracks, or a notice if the queue is empty.
    """
    if not interaction.guild:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    queue = music_queues[interaction.guild.id]
    if not queue:
        await interaction.response.send_message("The queue is empty. Use `/play` to add songs.", ephemeral=True)
        return

    embed = discord.Embed(title="Music Queue 🎵", color=discord.Color.blurple())
    for i, track in enumerate(queue, start=1):
        label = "Now Playing 🎵" if i == 1 else f"#{i}"
        embed.add_field(
            name=f"{label} — {track['title']}",
            value=f"Duration: {format_duration(track['duration'])} | Requested by: {track['requester']}",
            inline=False,
        )
    await interaction.response.send_message(embed=embed)


@tree.command(name="skip", description="Skip the current song")
async def skip(interaction: discord.Interaction) -> None:
    """
    This function; stops the current track and advances to the next in the queue.
    input: interaction - Discord interaction object.
    output: none. Confirms the skip, or reports that nothing is playing.
    """
    if not interaction.guild:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    vc = voice_clients.get(interaction.guild.id)
    if not vc or not vc.is_playing():
        await interaction.response.send_message("Nothing is playing right now.", ephemeral=True)
        return

    vc.stop() #Triggers after_playing, which pops the track and calls play_next.
    await interaction.response.send_message("Skipped ⏭️")
    log.info(f"[{interaction.user}] Skipped track in guild {interaction.guild.id}.")


@tree.command(name="leave-call", description="Make Axis leave the voice channel")
async def leave_call(interaction: discord.Interaction) -> None:
    """
    This function; disconnects the bot from the voice channel and clears the queue.
    input: interaction - Discord interaction object.
    output: none. Confirms the disconnect, or reports the bot is not in a channel.
    """
    if not interaction.guild:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    guild_id = interaction.guild.id
    vc = voice_clients.get(guild_id)
    if not vc or not vc.is_connected():
        await interaction.response.send_message("I am not in a voice channel.", ephemeral=True)
        return

    music_queues[guild_id].clear()
    await vc.disconnect()
    voice_clients.pop(guild_id, None)
    await interaction.response.send_message("Left the voice channel and cleared the queue. 👋")
    log.info(f"[{interaction.user}] Bot disconnected from guild {guild_id}.")


@tree.command(name="reset-queue", description="Clear the entire music queue")
async def reset_queue(interaction: discord.Interaction) -> None:
    """
    This function; stops playback and empties the queue without disconnecting the bot.
    input: interaction - Discord interaction object.
    output: none. Confirms the queue was cleared.
    """
    if not interaction.guild:
        await interaction.response.send_message("This command only works in a server.", ephemeral=True)
        return

    guild_id = interaction.guild.id
    vc = voice_clients.get(guild_id)
    if vc and vc.is_playing():
        vc.stop()
    music_queues[guild_id].clear()
    await interaction.response.send_message("Queue cleared. 🗑️")
    log.info(f"[{interaction.user}] Queue reset in guild {guild_id}.")

# =============================================
#  FUN
# =============================================
# ---------------------------------------------
#  Slash Commands (Fun)
# ---------------------------------------------
@tree.command(name="popularity", description="See who has chatted with Axis the most")
async def popularity(interaction: discord.Interaction) -> None:
    """
    This function; fetches the top chatters from the database and displays a leaderboard.
    input: interaction - Discord interaction object.
    output: none. Sends an embed leaderboard with display names, personalities, and message counts.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT user_id, current_personality, prompt_count FROM users "
            "ORDER BY prompt_count DESC LIMIT 10"
        )
        rows = await cursor.fetchall()

    if not rows:
        await interaction.response.send_message(
            "No chat history yet. Start talking with `/say`!", ephemeral=True
        )
        return

    medals = ["🥇", "🥈", "🥉"]
    embed = discord.Embed(title="🏆 Top Axis Chatters", color=discord.Color.gold())

    for i, (user_id, personality, count) in enumerate(rows):
        medal = medals[i] if i < 3 else f"#{i + 1}"
        member = interaction.guild.get_member(int(user_id)) if interaction.guild else None
        display = member.display_name if member else f"User {user_id}"
        embed.add_field(
            name=f"{medal} {display}",
            value=f"Personality: **{personality.capitalize()}** | Messages: **{count}**",
            inline=False,
        )
    await interaction.response.send_message(embed=embed)


@tree.command(name="help", description="See all available Axis commands")
async def help_command(interaction: discord.Interaction) -> None:
    """
    This function; sends a simple overview of every command grouped by category.
    input: interaction - Discord interaction object.
    output: none. Sends an ephemeral embed listing all commands with emoji labels.
    """
    embed = discord.Embed(
        title="Axis — Command Guide",
        description="Everything I can do, all in one place.",
        color=discord.Color.blurple(),
    )

    embed.add_field(name="🤖  AI Companion", value="\u200b", inline=False)
    embed.add_field(name="`/start-axis`", value="Set up your AI companion for the first time.", inline=False)
    embed.add_field(name="`/switch-axis`", value="Switch personality without losing your memories.", inline=False)
    embed.add_field(name="`/clear-axis`", value="Permanently wipe all your memories.", inline=False)
    embed.add_field(name="`/status-axis`", value="View your profile — personality, mood, memory, and more.", inline=False)
    embed.add_field(name="`/setname <name>`", value="Tell Axis your name so it never forgets.", inline=False)
    embed.add_field(name="`/say <message>`", value="Talk to Axis publicly in the channel.", inline=False)
    embed.add_field(name="`/whisper <message>`", value="Talk to Axis privately — response arrives via DM.", inline=False)

    embed.add_field(name="🎵  Music", value="\u200b", inline=False)
    embed.add_field(name="`/play <query or URL>`", value="Play a song by name or YouTube link. Queues automatically if something is already playing.", inline=False)
    embed.add_field(name="`/play-queue`", value="Show all songs coming up in the queue.", inline=False)
    embed.add_field(name="`/skip`", value="Skip the current song.", inline=False)
    embed.add_field(name="`/leave-call`", value="Make Axis leave the voice channel and clear the queue.", inline=False)
    embed.add_field(name="`/reset-queue`", value="Clear all songs without disconnecting.", inline=False)

    embed.add_field(name="🎉  Fun", value="\u200b", inline=False)
    embed.add_field(name="`/popularity`", value="Leaderboard of who has chatted with Axis the most.", inline=False)

    embed.set_footer(text="Use /start-axis to get started!")
    await interaction.response.send_message(embed=embed, ephemeral=True)

# =============================================
#  BOT LIFECYCLE
# =============================================
@client.event
async def on_ready() -> None:
    """
    This function; called once the bot connects to Discord. Sets up the database, syncs commands, and sets the status.
    input: none.
    output: none. Logs a ready message with the current configuration.
    """
    await init_db()
    await tree.sync()
    await client.change_presence(
        status=discord.Status.online,
        activity=discord.Activity(type=discord.ActivityType.listening, name="ya 💕"),
    )
    log.info(
        f"Axis is online as {client.user}  |  "
        f"concurrency={MAX_CONCURRENT}  |  "
        f"timeout={REQUEST_TIMEOUT}s  |  "
        f"memory_limit={MEMORY_LIMIT} exchanges/personality"
    )


client.run(TOKEN)