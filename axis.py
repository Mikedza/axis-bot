import discord
from discord import app_commands
import aiosqlite
import aiohttp
import asyncio
import json
import logging
from datetime import datetime

# ─────────────────────────────────────────────
#  Logging Setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                          # Console
        logging.FileHandler("axis.log", encoding="utf-8") # File
    ]
)
log = logging.getLogger("Axis")

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

MEMORY_LIMIT    = 8    # Conversation lines kept per personality
MAX_CONCURRENT  = 2    # Ollama simultaneous requests (raise carefully)
REQUEST_TIMEOUT = 120  # Seconds before an AI call times out

# ─────────────────────────────────────────────
#  Discord Client
# ─────────────────────────────────────────────
intents = discord.Intents.default()
client  = discord.Client(intents=intents)
tree    = app_commands.CommandTree(client)

# Semaphore: only MAX_CONCURRENT calls hit Ollama at once.
# The rest wait in line but DON'T block Discord's event loop.
ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# Live counter so we can report queue depth in logs
_queue_depth = 0

PERSONALITIES = {
    "gym": "You are 'Gym Bro'. You are a motivational male trainer. You are relatively toxic but deeply care about the user's progress.",
    "shy": "You are a 'Shy supportive girl'. You are quiet, sweet, and use cute emoticons like > - < or ^_^.",
    "dominant": "You are a 'Toxic yet kind dominant girl'. You are aggressive and commanding.",
    "bobu": "You are 'Bobu', a comedic Japanese friend born in Austin, Texas. You joke 90% of the time."
}

# ─────────────────────────────────────────────
#  Database Helpers
# ─────────────────────────────────────────────
async def init_db():
    async with aiosqlite.connect("database.db") as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            current_personality TEXT,
            memories TEXT,
            traits TEXT,
            mood TEXT
        )
        """)
        await db.commit()
    log.info("Database initialised.")

async def get_user(user_id):
    async with aiosqlite.connect("database.db") as db:
        cursor = await db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        return await cursor.fetchone()

async def save_user(user_id, personality, memories, traits, mood):
    async with aiosqlite.connect("database.db") as db:
        await db.execute(
            "INSERT OR REPLACE INTO users (user_id, current_personality, memories, traits, mood) VALUES (?, ?, ?, ?, ?)",
            (user_id, personality, json.dumps(memories), traits, mood)
        )
        await db.commit()

# ─────────────────────────────────────────────
#  AI Call
# ─────────────────────────────────────────────
async def ask_ai(prompt: str, user_tag: str) -> str:
    """
    Sends a prompt to the local Ollama instance.
    Wrapped in the semaphore by the caller, so this function
    just handles the HTTP work and its own error cases.
    """
    url     = "http://localhost:11434/api/generate"
    payload = {"model": "mistral-nemo", "prompt": prompt, "stream": False}
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    log.info(f"[{user_tag}] Sending prompt to Ollama ({len(prompt)} chars) …")
    t_start = asyncio.get_event_loop().time()

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                elapsed = asyncio.get_event_loop().time() - t_start
                if resp.status == 200:
                    data = await resp.json()
                    reply = data["response"]
                    log.info(f"[{user_tag}] Ollama replied in {elapsed:.1f}s ({len(reply)} chars).")
                    return reply
                else:
                    log.warning(f"[{user_tag}] Ollama returned HTTP {resp.status} after {elapsed:.1f}s.")
                    return f"System Error: AI service returned status {resp.status}."

    except asyncio.TimeoutError:
        log.error(f"[{user_tag}] Ollama timed out after {REQUEST_TIMEOUT}s.")
        return "System Error: The AI is taking too long to think. Please try again in a moment."
    except Exception as exc:
        log.error(f"[{user_tag}] Ollama connection error: {exc}")
        return f"System Error: Connection failed. Is Ollama running? ({exc})"

# ─────────────────────────────────────────────
#  UI Views
# ─────────────────────────────────────────────
class ConfirmClearView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=30)

    @discord.ui.button(label="Confirm Reset", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        async with aiosqlite.connect("database.db") as db:
            await db.execute("DELETE FROM users WHERE user_id = ?", (str(interaction.user.id),))
            await db.commit()
        log.info(f"[{interaction.user}] Memory wiped by user request.")
        await interaction.response.edit_message(content="Memory wiped. We are strangers now.", view=None)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="Reset cancelled.", view=None)


class PersonalityView(discord.ui.View):
    def __init__(self, user_id):
        super().__init__(timeout=60)
        self.user_id = user_id

    async def set_personality(self, interaction: discord.Interaction, personality: str):
        user = await get_user(str(self.user_id))
        memories = json.loads(user[2]) if user else {}
        traits   = user[3] if user else ""
        mood     = user[4] if user else "Neutral"

        if personality not in memories:
            memories[personality] = ""

        await save_user(str(self.user_id), personality, memories, traits, mood)
        log.info(f"[{interaction.user}] Personality set to '{personality}'.")
        await interaction.response.edit_message(
            content=f"Personality set to **{personality.capitalize()}**.",
            view=None
        )

    @discord.ui.button(label="Gym Bro",      style=discord.ButtonStyle.primary)
    async def gym(self, i: discord.Interaction, b: discord.ui.Button):
        await self.set_personality(i, "gym")

    @discord.ui.button(label="Shy Support",  style=discord.ButtonStyle.success)
    async def shy(self, i: discord.Interaction, b: discord.ui.Button):
        await self.set_personality(i, "shy")

    @discord.ui.button(label="Dominant",     style=discord.ButtonStyle.danger)
    async def dominant(self, i: discord.Interaction, b: discord.ui.Button):
        await self.set_personality(i, "dominant")

    @discord.ui.button(label="Bobu",         style=discord.ButtonStyle.secondary)
    async def bobu(self, i: discord.Interaction, b: discord.ui.Button):
        await self.set_personality(i, "bobu")

# ─────────────────────────────────────────────
#  Slash Commands
# ─────────────────────────────────────────────
@tree.command(name="start-axis", description="Initialize your AI companion")
async def start_axis(interaction: discord.Interaction):
    user = await get_user(str(interaction.user.id))
    if user:
        await interaction.response.send_message(
            "You already have memories! Use `/clear-axis` to restart.", ephemeral=True
        )
        return
    log.info(f"[{interaction.user}] Starting Axis setup.")
    await interaction.response.send_message(
        "Who would you like to talk to?",
        view=PersonalityView(interaction.user.id),
        ephemeral=True
    )

@tree.command(name="clear-axis", description="Permanently delete your memory")
async def clear_axis(interaction: discord.Interaction):
    log.info(f"[{interaction.user}] Requested memory clear.")
    await interaction.response.send_message("Are you sure?", view=ConfirmClearView(), ephemeral=True)

# ─────────────────────────────────────────────
#  Core Chat Handler
# ─────────────────────────────────────────────
async def handle_chat(interaction: discord.Interaction, prompt: str, private: bool = False):
    global _queue_depth
    user_tag = str(interaction.user)

    try:
        # ✅ ALWAYS defer first
        await interaction.response.defer(ephemeral=private)

        user = await get_user(str(interaction.user.id))
        if not user:
            await interaction.followup.send(
                "Please use `/start-axis` first!",
                ephemeral=True
            )
            return

        personality = user[1]

        # ✅ Safe JSON load
        try:
            memories = json.loads(user[2]) if user[2] else {}
        except:
            memories = {}

        traits  = user[3] or "None"
        mood    = user[4] or "Neutral"
        history = memories.get(personality, "")

        full_prompt = (
            f"Instruction: {PERSONALITIES[personality]}\n"
            f"Traits: {traits}\n"
            f"Mood: {mood}\n"
            f"History: {history}\n"
            f"User: {prompt}\n"
            f"Assistant:"
        )

        # Queue tracking
        _queue_depth += 1

        async with ai_semaphore:
            response = await ask_ai(full_prompt, user_tag)

        _queue_depth -= 1

        # Save memory
        new_history = history + f"\nUser: {prompt}\nAssistant: {response}"
        lines = new_history.split("\n")
        memories[personality] = "\n".join(lines[-MEMORY_LIMIT:])

        await save_user(str(interaction.user.id), personality, memories, traits, mood)

        # ✅ Send response (ONLY HERE)
        if private:
            try:
                dm = await interaction.user.create_dm()
                await dm.send(response)

                await interaction.followup.send(
                    "📩 Check your DMs!",
                    ephemeral=True
                )
            except discord.Forbidden:
                await interaction.followup.send(
                    "I can't DM you. Enable DMs from server members.",
                    ephemeral=True
                )
        else:
            await interaction.followup.send(response)

    except Exception:
        log.exception(f"[{user_tag}] CRASH in handle_chat")

        if interaction.response.is_done():
            await interaction.followup.send(
                "⚠️ Something broke internally.",
                ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "⚠️ Something broke internally.",
                ephemeral=True
            )
@tree.command(name="say",     description="Talk to Axis publicly")
async def say(interaction: discord.Interaction, prompt: str):
    await handle_chat(interaction, prompt, private=False)

@tree.command(name="whisper", description="Talk to Axis in DMs")
async def whisper(interaction: discord.Interaction, prompt: str):
    await handle_chat(interaction, prompt, private=True)

# ─────────────────────────────────────────────
#  Bot Lifecycle
# ─────────────────────────────────────────────
@client.event
async def on_ready():
    await init_db()
    await tree.sync()
    log.info(f"Axis is online as {client.user}  |  concurrency={MAX_CONCURRENT}  |  timeout={REQUEST_TIMEOUT}s")

client.run(TOKEN)