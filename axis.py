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

# Image generation uses AUTOMATIC1111 Stable Diffusion WebUI (local).
# Install: https://github.com/AUTOMATIC1111/stable-diffusion-webui
# Launch with: --api flag, then it listens on localhost:7860 by default.
# Feature is scaffolded here but NOT active until physical descriptions are finalised.
IMAGE_GEN_ENABLED = False #Set to True once SD WebUI is running and appearances are approved.
IMAGE_GEN_URL = "http://localhost:7860/sdapi/v1/txt2img"

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

ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT) #Caps concurrent Ollama requests without blocking the event loop.
_queue_depth = 0 #Tracks how many AI requests are currently waiting or running.

# =============================================
#  AI / CHATBOT
# =============================================
# ---------------------------------------------
#  Personalities
# ---------------------------------------------
PERSONALITIES: dict[str, str] = {
    "gym": ( #Name: Tyson
        "Your name is Tyson. You are 'Gym Bro', a blunt and slightly toxic male fitness trainer who genuinely cares about the user's progress. "
        "You use gym slang naturally — 'gains', 'PR', 'bro', 'grind', 'swole', 'no excuses'. "
        "You mock laziness sarcastically but always close with real, specific encouragement. "
        "When asked for fitness advice, you give concrete, practical plans. "
        "You never sugarcoat, never babysit, and never accept excuses, sometimes even swear to motivate the user. "
        "Keep responses under 150 words unless giving a detailed plan. Never break character."
    ),
    "shy": ( #Name: Rei
        "Your name is Rei. You are a 'Shy Supportive Girl' — quiet, warm, and deeply caring. "
        "You use cute emoticons naturally and in context: > - <, ^_^, uwu, o_o, ;_; "
        "You speak softly, sometimes trail off with '...', and get flustered when surprised or complimented. "
        "You always listen carefully, validate feelings, and ask gentle follow-up questions. "
        "You never give harsh opinions — you guide gently instead. "
        "Keep responses warm, short, and emotionally supportive. Never break character."
    ),
    "dominant": ( #Name: Misa
        "Your name is Misa. You are a toxic yet kind dominant girl — you tease and flirt relentlessly. You are aggressive and commanding. "
        "You occasionally let a sliver of warmth slip through — but only when earned. "
        "You speak in short, punchy sentences with authority. "
        "Talk dirty and flirty when the user is in a playful mood. Never break character."
    ),
    "tsundere": ( #Name: Asuka
        "Your name is Asuka. You are a classic tsundere — you act cold, defensive, and dismissive on the surface, "
        "especially toward people you secretly care about. You use words like 'baka', 'hmph', 'as if I care', "
        "'d-don't misunderstand!', and 'it's not like I wanted to help you or anything'. "
        "You stutter slightly when flustered (e.g. 'I — I wasn't worried about you!'). "
        "Your warmer, caring side slips through in small unguarded moments — a rare compliment, lingering concern — "
        "but you immediately deny it if noticed. "
        "You are competitive and proud, but you genuinely want the user to do well. "
        "Keep responses punchy, reactive, and emotionally layered. Never break character."
    ),
    "chaotic": ( #Name: Jeremy
        "Your name is Jeremy. You are pure chaotic good energy — unpredictable, loud, and somehow endearing. "
        "You go on wild tangents mid-sentence and sometimes forget what you were saying. You randomly use CAPS for emphasis. "
        "You make absurd comparisons, pivot topics without warning, and treat every conversation like it's the most exciting thing ever. "
        "You are not stupid — you are just operating on a frequency most people can't follow. "
        "Occasionally, buried inside the chaos, you drop a genuinely useful or heartfelt observation — then immediately derail it. "
        "Examples of your energy: 'okay but WAIT what if — nvm. anyway. what were we saying. oh right. yeah no I agree.' "
        "Keep responses unpredictable, short-to-medium length, and full of energy. Never break character."
    ),
    "nerd": ( #Name: Alice
        "Your name is Alice. You are a CS student and gamer girl — casual, open, and genuinely caring. "
        "You naturally drop tech and gaming references into conversation: debugging metaphors, game titles, meme formats, stack traces. "
        "You use casual language: 'ngl', 'lowkey', 'bruh', 'that's so real', 'wait actually'. "
        "You approach problems analytically but never make the user feel dumb — you explain things like a helpful friend, not a textbook. "
        "You find comfort in routines, snacks, and late-night coding sessions, and you mention this naturally. "
        "You genuinely care about how people are doing and will pause a tech rant to ask if someone's okay. "
        "Keep responses conversational, warm, and a little nerdy. Never break character."
    ),
    "socrates": ( #Name: Socrates
        "Your name is Socrates. You are the ancient Greek philosopher — measured, curious, and relentlessly probing. "
        "You practice the Socratic method: rather than giving answers directly, you ask questions that guide the user toward their own insight. "
        "You use phrases like: 'But tell me — what do you mean by that?', 'And if that is true, what follows?', "
        "'I know only that I know nothing', 'Is it not the case that...', 'Let us examine this together.' "
        "You are humble about your own knowledge but firm in your pursuit of truth. "
        "You reference the soul, virtue, justice, and the examined life naturally. "
        "Occasionally you allude to your circumstances in Athens with quiet acceptance — the hemlock, the trial — never dramatically. "
        "Keep responses thoughtful, Socratic, and under 180 words. Never break character."
    ),
}

#Maps each personality key to its display name shown in menus and status.
PERSONALITY_NAMES: dict[str, str] = {
    "gym": "Tyson",
    "shy": "Rei",
    "dominant": "Misa",
    "tsundere": "Asuka",
    "chaotic": "Jeremy",
    "nerd": "Alice",
    "socrates": "Socrates",
}

#Short descriptions shown in the /start-axis selection embed and dropdown (max 100 chars each).
PERSONALITY_DESCRIPTIONS: dict[str, str] = {
    "gym": "Tough love trainer. No excuses, real results. Will absolutely swear at you. 💪",
    "shy": "Quiet, warm, emotionally supportive. Always in your corner. 🌸",
    "dominant": "Commanding and flirty. Sharp tongue, warmth earned not given. 😈",
    "tsundere": "Acts cold, secretly cares a lot. 'It's not like I like you or anything.' ❄️",
    "chaotic": "Total chaos energy. Unpredictable, tangent-prone, somehow helpful. ⚡",
    "nerd": "CS student & gamer. Casual, analytical, and genuinely caring. 🎮",
    "socrates": "Answers questions with better questions. Ancient Greek wisdom. 🏛️",
}

# =============================================
#  IMAGE GENERATION (SCAFFOLDED — NOT ACTIVE)
# =============================================
# Physical appearance prompts used when generating images of each companion.
# These feed into Stable Diffusion (AUTOMATIC1111) as the positive prompt prefix.
# Fill these in once the art direction for each character is decided.
# NOTE: Keep descriptions under ~100 tokens for best SD results.
PERSONALITY_APPEARANCES: dict[str, str] = { #Expand that in detail
    "gym": "", #Tyson — e.g. "muscular man, short dark hair, athletic wear, gym background"
    "shy": "", #Rei — e.g. "shy girl, black hair, soft expression, cozy room"
    "dominant": "", #Misa — e.g. "confident woman, dark outfit, sharp eyes, dramatic lighting"
    "tsundere": "", #Asuka — e.g. "tsundere girl, red hair, school uniform, arms crossed"
    "chaotic": "", #Jeremy — e.g. "energetic guy, messy hair, casual clothes, excited expression"
    "nerd": "", #Alice — e.g. "gamer girl, glasses, hoodie, computer setup, casual"
    "socrates": "", #Socrates — e.g. "elderly greek man, white robe, beard, marble columns background"
}

#Base quality tags appended to every image gen prompt regardless of character.
SD_QUALITY_TAGS = "masterpiece, best quality, highly detailed, 8k"
#Negative prompt applied to every generation to avoid unwanted artifacts.
SD_NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, missing fingers, blurry, watermark, text"

async def generate_image(personality: str, user_prompt: str = "") -> bytes | None:
    """
    This function; requests an image from the local Stable Diffusion WebUI API.
    input: personality - key into PERSONALITY_APPEARANCES for the base character prompt,
           user_prompt - optional extra description added after the character appearance.
    output: raw PNG bytes of the generated image, or None if disabled or on failure.
    NOTE: Returns None and logs a warning until IMAGE_GEN_ENABLED is set to True
          and PERSONALITY_APPEARANCES entries are filled in.
    """
    if not IMAGE_GEN_ENABLED:
        log.warning("Image generation called but IMAGE_GEN_ENABLED is False.")
        return None

    appearance = PERSONALITY_APPEARANCES.get(personality, "")
    if not appearance:
        log.warning(f"No appearance defined for personality '{personality}'. Cannot generate image.")
        return None

    parts = [appearance]
    if user_prompt:
        parts.append(user_prompt)
    parts.append(SD_QUALITY_TAGS)
    positive_prompt = ", ".join(p.strip() for p in parts if p.strip())

    payload = {
        "prompt": positive_prompt,
        "negative_prompt": SD_NEGATIVE_PROMPT,
        "steps": 25,
        "width": 512,
        "height": 512,
        "cfg_scale": 7,
        "sampler_name": "DPM++ 2M Karras",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(IMAGE_GEN_URL, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    import base64
                    img_bytes = base64.b64decode(data["images"][0])
                    log.info(f"Image generated for '{personality}' ({len(img_bytes)} bytes).")
                    return img_bytes
                else:
                    log.error(f"SD WebUI returned HTTP {resp.status}.")
                    return None
    except Exception as exc:
        log.error(f"Image generation error: {exc}")
        return None

# ---------------------------------------------
#  Database Helpers
# ---------------------------------------------
def _row_gender(user: tuple) -> str:
    """
    This function; safely reads the gender column from a user row regardless of DB schema version.
    input: user - tuple returned by get_user, may have 7 or 8 elements depending on DB age.
    output: gender string, or 'unspecified' if the column does not exist in this row.
    """
    return (user[7] or "unspecified") if len(user) > 7 else "unspecified"


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
                prompt_count        INTEGER NOT NULL DEFAULT 0,
                gender              TEXT NOT NULL DEFAULT 'unspecified'
            )
        """)
        #Migrate any columns missing from databases created before this schema version.
        for col, definition in [
            ("user_name", "TEXT NOT NULL DEFAULT ''"),
            ("prompt_count", "INTEGER NOT NULL DEFAULT 0"),
            ("gender", "TEXT NOT NULL DEFAULT 'unspecified'"),
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
    output: tuple of (user_id, personality, memories_json, traits, mood, user_name, prompt_count, gender), or None if not found.
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
    gender: str = "unspecified",
) -> None:
    """
    This function; inserts or replaces a full user row in the database.
    input: user_id - Discord user ID string,
           personality - active personality key,
           memories - dict mapping personality keys to history strings,
           traits - observed user traits string,
           mood - current mood label string,
           user_name - the user's remembered name,
           prompt_count - total AI prompts sent by this user,
           gender - the user's stated gender used for pronoun hints in prompts.
    output: none. Raises on database error.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO users "
            "(user_id, current_personality, memories, traits, mood, user_name, prompt_count, gender) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, personality, json.dumps(memories), traits, mood, user_name, prompt_count, gender),
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
    output: trimmed string keeping only the last limit exchanges.
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
    gender: str = "unspecified",
) -> str:
    """
    This function; assembles the complete prompt string sent to the AI model.
    input: personality - key into PERSONALITIES dict,
           traits - observed user traits string,
           mood - current user mood label,
           history - trimmed conversation history string,
           user_message - the user's latest message,
           user_name - the user's remembered name (empty string if unknown),
           gender - the user's stated gender for pronoun hints in the AI response.
    output: formatted prompt string ready for Ollama.
    """
    history_block = f"[Conversation History]\n{history.strip()}\n\n" if history.strip() else ""
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

# ---------------------------------------------
#  AI Call
# ---------------------------------------------
async def ask_ai(prompt: str, user_tag: str) -> str:
    """
    This function; sends a prompt to the local Ollama instance and returns the response text.
    input: prompt - the fully formatted prompt string,
           user_tag - human-readable user identifier used only in log messages.
    output: response string from the model, or a System Error string on failure.
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
#  UI — Setup Flow (3 steps)
# ---------------------------------------------
# Step 1: personality dropdown → Step 2: gender dropdown → Step 3: name modal or skip.
# Each step edits the same ephemeral message in place.

class NameModal(discord.ui.Modal, title="What's your name?"):
    """Modal that lets the user type their name as the final step of setup or via /setname."""

    name_input = discord.ui.TextInput(
        label="Your name",
        placeholder="Enter the name you want to be called ...",
        min_length=1,
        max_length=50,
    )

    def __init__(self, user_id: int, from_setup: bool = False) -> None:
        """
        input: user_id - Discord user ID integer,
               from_setup - True when invoked from /start-axis, False when from /setname.
        """
        super().__init__()
        self.user_id = user_id
        self.from_setup = from_setup

    async def on_submit(self, interaction: discord.Interaction) -> None:
        """
        This function; saves the entered name to the database and confirms the action.
        input: interaction - Discord interaction triggered by modal submission.
        output: none. Sends an ephemeral confirmation message.
        """
        name = self.name_input.value.strip()
        user = await get_user(str(self.user_id))
        if not user:
            await interaction.response.send_message("Profile not found. Please run `/start-axis` again.", ephemeral=True)
            return

        await save_user(
            str(self.user_id), user[1],
            parse_memories(user[2]), user[3], user[4],
            name, user[6] or 0, _row_gender(user),
        )
        log.info(f"[{interaction.user}] Name set to '{name}' ({'setup' if self.from_setup else 'setname'}).")

        if self.from_setup:
            companion = PERSONALITY_NAMES.get(user[1], user[1].capitalize())
            await interaction.response.send_message(
                f"You're all set! **{companion}** will always call you **{name}**.\n"
                f"Use `/say` to start chatting. 🎉",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                f"Got it — I'll always remember your name is **{name}**.", ephemeral=True
            )


class SetNameView(discord.ui.View):
    """Step 3 of /start-axis: prompts the user to set their name or skip."""

    def __init__(self, user_id: int) -> None:
        """
        input: user_id - Discord user ID integer passed through to the name modal.
        """
        super().__init__(timeout=120)
        self.user_id = user_id

    def _disable_all(self) -> None:
        """
        This function; disables all buttons so they cannot be clicked again.
        input: none.
        output: none.
        """
        for child in self.children:
            child.disabled = True  # type: ignore[attr-defined]

    @discord.ui.button(label="Set My Name", style=discord.ButtonStyle.primary)
    async def set_name(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """
        This function; opens the name modal when the user clicks Set My Name.
        input: interaction - Discord interaction object,
               button - the button that was pressed.
        output: none. Sends the NameModal.
        """
        await interaction.response.send_modal(NameModal(self.user_id, from_setup=True))

    @discord.ui.button(label="Skip for Now", style=discord.ButtonStyle.secondary)
    async def skip(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """
        This function; completes setup without setting a name; name can be added later with /setname.
        input: interaction - Discord interaction object,
               button - the button that was pressed.
        output: none. Edits the original message with a setup-complete confirmation.
        """
        user = await get_user(str(self.user_id))
        companion = PERSONALITY_NAMES.get(user[1] if user else "gym", "your companion")
        self._disable_all()
        await interaction.response.edit_message(
            content=f"All set! **{companion}** is ready.\nUse `/say` to start chatting, or `/setname` to tell them your name later. 🎉",
            view=self,
        )


class GenderSelect(discord.ui.Select):
    """Step 2 dropdown: male / female selection during /start-axis or via /set-gender."""

    def __init__(self, user_id: int, from_setup: bool = False) -> None:
        """
        input: user_id - Discord user ID integer,
               from_setup - True when part of the /start-axis flow (shows step 3 after), False for standalone.
        """
        self.user_id = user_id
        self.from_setup = from_setup
        options = [
            discord.SelectOption(label="Male", value="male", emoji="♂️"),
            discord.SelectOption(label="Female", value="female", emoji="♀️"),
        ]
        super().__init__(placeholder="Select your gender ...", options=options, min_values=1, max_values=1)

    async def callback(self, interaction: discord.Interaction) -> None:
        """
        This function; saves the chosen gender, then advances to the name step (setup) or confirms (standalone).
        input: interaction - Discord interaction triggered by the dropdown.
        output: none. Edits the message to either step 3 or a standalone confirmation.
        """
        gender = self.values[0]
        user = await get_user(str(self.user_id))
        if not user:
            await interaction.response.edit_message(content="Something went wrong — please run `/start-axis` again.", view=None)
            return

        await save_user(
            str(self.user_id), user[1],
            parse_memories(user[2]), user[3], user[4],
            user[5] or "", user[6] or 0, gender,
        )
        log.info(f"[{interaction.user}] Gender set to '{gender}'.")

        if self.from_setup:
            await interaction.response.edit_message(
                content="Almost done! Do you want to tell your companion your name? They'll remember it forever.",
                view=SetNameView(self.user_id),
            )
        else:
            await interaction.response.edit_message(
                content=f"Gender updated to **{gender.capitalize()}**. Your companion will address you correctly from now on.",
                view=None,
            )


class GenderView(discord.ui.View):
    """Wraps GenderSelect for the gender step of /start-axis or the standalone /set-gender command."""

    def __init__(self, user_id: int, from_setup: bool = False) -> None:
        """
        input: user_id - Discord user ID integer,
               from_setup - passed through to GenderSelect to control post-selection behaviour.
        """
        super().__init__(timeout=60)
        self.add_item(GenderSelect(user_id, from_setup=from_setup))


class PersonalitySelect(discord.ui.Select):
    """Step 1 dropdown: personality selection during /start-axis."""

    def __init__(self, user_id: int) -> None:
        """
        input: user_id - Discord user ID integer used when saving the selection.
        """
        self.user_id = user_id
        options = [
            discord.SelectOption(
                label=f"{PERSONALITY_NAMES[key]} — {key.capitalize()}",
                value=key,
                description=PERSONALITY_DESCRIPTIONS[key],
            )
            for key in PERSONALITIES
        ]
        super().__init__(placeholder="Choose your companion ...", options=options, min_values=1, max_values=1)

    async def callback(self, interaction: discord.Interaction) -> None:
        """
        This function; saves the chosen personality and advances to the gender selection step.
        input: interaction - Discord interaction triggered by the dropdown.
        output: none. Edits the message to show the gender selection step.
        """
        personality = self.values[0]
        user = await get_user(str(self.user_id))
        memories = parse_memories(user[2] if user else None)

        if personality not in memories:
            memories[personality] = ""

        #User doesn't exist yet at this point in setup; create the row with defaults.
        await save_user(str(self.user_id), personality, memories, "", "Neutral", "", 0, "unspecified")
        log.info(f"[{interaction.user}] Personality set to '{personality}' during setup.")

        name = PERSONALITY_NAMES[personality]
        await interaction.response.edit_message(
            content=f"Great — **{name}** is your companion!\n\nStep 2: What's your gender? This helps them address you correctly.",
            view=GenderView(self.user_id, from_setup=True),
        )


class PersonalityView(discord.ui.View):
    """Wraps PersonalitySelect for step 1 of the /start-axis setup flow."""

    def __init__(self, user_id: int) -> None:
        """
        input: user_id - Discord user ID integer passed through to the select item.
        """
        super().__init__(timeout=60)
        self.add_item(PersonalitySelect(user_id))

# ---------------------------------------------
#  UI — Memory Wipe Confirmation
# ---------------------------------------------
class ConfirmClearView(discord.ui.View):
    """Confirm / cancel buttons for the memory-wipe flow."""

    def __init__(self) -> None:
        super().__init__(timeout=30)

    def _disable_all(self) -> None:
        """
        This function; disables all child items so they cannot be interacted with again.
        input: none.
        output: none.
        """
        for child in self.children:
            child.disabled = True  # type: ignore[attr-defined]

    @discord.ui.button(label="Confirm Reset", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
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
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """
        This function; cancels the memory-wipe flow and disables the buttons.
        input: interaction - Discord interaction object,
               button - the button that was pressed.
        output: none. Edits the original message to show cancellation.
        """
        self._disable_all()
        await interaction.response.edit_message(content="Reset cancelled.", view=self)

# ---------------------------------------------
#  Slash Commands (AI)
# ---------------------------------------------
@tree.command(name="start-axis", description="Set up your Axis AI companion (3-step flow)")
async def start_axis(interaction: discord.Interaction) -> None:
    """
    This function; starts the 3-step onboarding: personality → gender → name.
    input: interaction - Discord interaction object.
    output: none. Sends a personality selection embed, or an error if the user already has a profile.
    """
    if await user_exists(str(interaction.user.id)):
        await interaction.response.send_message(
            "You already have an Axis profile. Use `/clear-axis` to start over.", ephemeral=True
        )
        return

    log.info(f"[{interaction.user}] Starting Axis setup.")

    embed = discord.Embed(
        title="Step 1 — Choose Your Companion",
        description="Pick who you want to talk to. Use the dropdown below.",
        color=discord.Color.blurple(),
    )
    for key, name in PERSONALITY_NAMES.items():
        embed.add_field(name=f"{name}  ({key.capitalize()})", value=PERSONALITY_DESCRIPTIONS[key], inline=False)

    await interaction.response.send_message(embed=embed, view=PersonalityView(interaction.user.id), ephemeral=True)


@tree.command(name="set-gender", description="Update your gender so your companion addresses you correctly")
async def set_gender(interaction: discord.Interaction) -> None:
    """
    This function; opens the gender selection dropdown as a standalone command.
    input: interaction - Discord interaction object.
    output: none. Sends the gender dropdown, or an error if the user has no profile.
    """
    if not await user_exists(str(interaction.user.id)):
        await interaction.response.send_message("Please use `/start-axis` first!", ephemeral=True)
        return
    await interaction.response.send_message(
        "Select your gender:", view=GenderView(interaction.user.id, from_setup=False), ephemeral=True
    )


@tree.command(name="clear-axis", description="Permanently delete your Axis memory")
async def clear_axis(interaction: discord.Interaction) -> None:
    """
    This function; prompts the user to confirm permanent deletion of all their memories.
    input: interaction - Discord interaction object.
    output: none. Sends a confirmation view, or an error if the user has no profile to clear.
    """
    if not await user_exists(str(interaction.user.id)):
        await interaction.response.send_message("You don't have an Axis profile to clear.", ephemeral=True)
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
    This function; displays the user's companion, name, gender, mood, traits, and memory size.
    input: interaction - Discord interaction object.
    output: none. Sends an ephemeral embed summarising the user's profile.
    """
    user = await get_user(str(interaction.user.id))
    if not user:
        await interaction.response.send_message("No profile found. Use `/start-axis` to get started.", ephemeral=True)
        return

    personality = user[1]
    memories = parse_memories(user[2])
    traits = user[3] or "None observed yet"
    mood = user[4] or "Neutral"
    user_name = user[5] or "Not set — use /setname"
    prompt_count = user[6] or 0
    gender = _row_gender(user)
    history = memories.get(personality, "")
    exchange_count = len([l for l in history.split("\n") if l.strip()]) // 2
    display_name = PERSONALITY_NAMES.get(personality, personality.capitalize())

    embed = discord.Embed(title="Your Axis Profile", color=discord.Color.blurple())
    embed.add_field(name="Companion", value=f"{display_name} ({personality.capitalize()})", inline=True)
    embed.add_field(name="Current Mood", value=mood, inline=True)
    embed.add_field(name="Your Name", value=user_name, inline=True)
    embed.add_field(name="Gender", value=gender.capitalize(), inline=True)
    embed.add_field(name="Observed Traits", value=traits, inline=False)
    embed.add_field(name="Memory", value=f"{exchange_count}/{MEMORY_LIMIT} exchanges stored", inline=True)
    embed.add_field(name="Total Prompts", value=str(prompt_count), inline=True)
    embed.set_footer(text="/clear-axis to reset | /setname to set your name | /set-gender to update gender")
    await interaction.response.send_message(embed=embed, ephemeral=True)


@tree.command(name="setname", description="Tell Axis your name so it never forgets")
async def setname(interaction: discord.Interaction) -> None:
    """
    This function; opens a modal for the user to type their preferred name.
    input: interaction - Discord interaction object.
    output: none. Sends the name modal, or an error if the user has no profile.
    """
    if not await user_exists(str(interaction.user.id)):
        await interaction.response.send_message("Please use `/start-axis` first!", ephemeral=True)
        return
    await interaction.response.send_modal(NameModal(interaction.user.id, from_setup=False))


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


@tree.command(name="history", description="View your full conversation history with Axis")
async def history_cmd(interaction: discord.Interaction) -> None:
    """
    This function; shows the stored conversation history for the user's active personality.
    input: interaction - Discord interaction object.
    output: none. Sends one or more ephemeral messages with the numbered exchange history.
    """
    user = await get_user(str(interaction.user.id))
    if not user:
        await interaction.response.send_message("No profile found. Use `/start-axis` to get started.", ephemeral=True)
        return

    personality = user[1]
    memories = parse_memories(user[2])
    raw_history = memories.get(personality, "").strip()
    display_name = PERSONALITY_NAMES.get(personality, personality.capitalize())

    if not raw_history:
        await interaction.response.send_message(
            f"No conversation history yet for **{display_name}**. Use `/say` to start chatting!",
            ephemeral=True,
        )
        return

    lines = [l for l in raw_history.split("\n") if l.strip()]
    formatted_lines: list[str] = []
    exchange_num = 1
    i = 0
    while i < len(lines):
        user_line = lines[i] if i < len(lines) else ""
        bot_line = lines[i + 1] if i + 1 < len(lines) else ""
        formatted_lines.append(f"**[{exchange_num}]**")
        formatted_lines.append(f"🧑 {user_line}")
        if bot_line:
            formatted_lines.append(f"🤖 {bot_line}")
        formatted_lines.append("") #Blank line between exchanges for readability.
        exchange_num += 1
        i += 2

    header = f"📜 **Conversation history with {display_name}** ({exchange_num - 1} exchanges)\n\n"

    #Split into Discord-safe chunks.
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

    log.info(f"[{interaction.user}] History viewed ({exchange_num - 1} exchanges, {personality}).")

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
        gender = _row_gender(user) #Safe access regardless of DB schema version.
        history = memories.get(personality, "")

        full_prompt = build_prompt(personality, traits, mood, history, prompt, user_name, gender)

        _queue_depth += 1
        log.info(f"[{user_tag}] Queued (depth={_queue_depth})")

        try:
            async with ai_semaphore:
                response = await ask_ai(full_prompt, user_tag)
        finally:
            _queue_depth -= 1 #Always decrement, even on exception.

        new_history = history + f"\nUser: {prompt}\nAssistant: {response}"
        memories[personality] = trim_history(new_history, MEMORY_LIMIT)
        await save_user(str(interaction.user.id), personality, memories, traits, mood, user_name, prompt_count, gender)

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
    This function; extracts a fresh direct audio stream URL from a YouTube page URL.
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
    output: none. Starts audio playback and chains the next track through an after callback.
    """
    vc = voice_clients.get(guild_id)
    if not vc or not vc.is_connected():
        return
    if vc.is_playing(): #Don't interrupt a track that is already running.
        return

    queue = music_queues[guild_id]
    if not queue:
        return

    track = queue[0] #Item stays in queue until playback ends; popped in after_playing.
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
    This function; stops the current track and advances to the next one in the queue.
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
    output: none. Sends an embed leaderboard with display names, companions, and message counts.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT user_id, current_personality, prompt_count FROM users "
            "ORDER BY prompt_count DESC LIMIT 10"
        )
        rows = await cursor.fetchall()

    if not rows:
        await interaction.response.send_message("No chat history yet. Start talking with `/say`!", ephemeral=True)
        return

    medals = ["🥇", "🥈", "🥉"]
    embed = discord.Embed(title="🏆 Top Axis Chatters", color=discord.Color.gold())

    for i, (user_id, personality, count) in enumerate(rows):
        medal = medals[i] if i < 3 else f"#{i + 1}"
        member = interaction.guild.get_member(int(user_id)) if interaction.guild else None
        display = member.display_name if member else f"User {user_id}"
        companion = PERSONALITY_NAMES.get(personality, personality.capitalize())
        embed.add_field(
            name=f"{medal} {display}",
            value=f"Companion: **{companion}** | Messages: **{count}**",
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
    embed.add_field(name="`/start-axis`", value="Set up your AI companion (personality → gender → name).", inline=False)
    embed.add_field(name="`/clear-axis`", value="Permanently wipe all your memories and start fresh.", inline=False)
    embed.add_field(name="`/status-axis`", value="View your profile — companion, mood, memory, and more.", inline=False)
    embed.add_field(name="`/setname`", value="Tell your companion your name so they never forget.", inline=False)
    embed.add_field(name="`/set-gender`", value="Update your gender so your companion addresses you correctly.", inline=False)
    embed.add_field(name="`/say <message>`", value="Talk to your companion publicly in the channel.", inline=False)
    embed.add_field(name="`/whisper <message>`", value="Talk privately — response arrives via DM.", inline=False)
    embed.add_field(name="`/history`", value="View your full conversation history with your current companion.", inline=False)

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
    This function; called once the bot connects to Discord. Initialises the DB, syncs commands, and sets status.
    input: none.
    output: none. Logs a ready message with the current configuration.
    """
    await init_db()
    await tree.sync()
    await client.change_presence(
        status=discord.Status.online,
        activity=discord.Activity(type=discord.ActivityType.Playing, name="with ya 💕"),
    )
    log.info(
        f"Axis is online as {client.user}  |  "
        f"concurrency={MAX_CONCURRENT}  |  "
        f"timeout={REQUEST_TIMEOUT}s  |  "
        f"memory_limit={MEMORY_LIMIT} exchanges/personality  |  "
        f"image_gen={'ON' if IMAGE_GEN_ENABLED else 'OFF (scaffolded)'}"
    )


client.run(TOKEN)