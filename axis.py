import discord
from discord import app_commands
import aiosqlite
import aiohttp  # Replaced requests with aiohttp
import asyncio
import json

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

MEMORY_LIMIT = 8
# This limits the AI to processing 1 request at a time to prevent Ollama from crashing.
ai_semaphore = asyncio.Semaphore(1) 

PERSONALITIES = {
    "gym": "You are 'Gym Bro'. You are a motivational male trainer. You are relatively toxic but deeply care about the user's progress.",
    "shy": "You are a 'Shy supportive girl'. You are quiet, sweet, and use cute emoticons like > - < or ^_^.",
    "dominant": "You are a 'Toxic yet kind dominant girl'. You are aggressive and commanding.",
    "bobu": "You are 'Bobu', a comedic Japanese friend born in Austin, Texas. You joke 90% of the time."
}

"""
This function; initializes the database.
input: None.
output: None.
"""
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

"""
This function; retrieves a user's data.
input: user_id (string).
output: user (tuple) or None.
"""
async def get_user(user_id):
    async with aiosqlite.connect("database.db") as db:
        cursor = await db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        return await cursor.fetchone()

"""
This function; saves user data.
input: user_id (string), personality (string), memories (dict), traits (string), mood (string).
output: None.
"""
async def save_user(user_id, personality, memories, traits, mood):
    async with aiosqlite.connect("database.db") as db:
        await db.execute(
            "INSERT OR REPLACE INTO users (user_id, current_personality, memories, traits, mood) VALUES (?, ?, ?, ?, ?)",
            (user_id, personality, json.dumps(memories), traits, mood)
        )
        await db.commit()

"""
This function; calls the local Ollama API asynchronously.
input: prompt (string).
output: response (string).
"""
async def ask_ai(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral-nemo",
        "prompt": prompt,
        "stream": False
    }
    
    # Using aiohttp for non-blocking requests
    # Increased timeout to 120 seconds to allow for queuing
    timeout = aiohttp.ClientTimeout(total=120) 
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["response"]
                else:
                    return f"System Error: AI service returned status {response.status}."
    except asyncio.TimeoutError:
        return "System Error: The AI is taking too long to think. Please try again in a moment."
    except Exception as e:
        return f"System Error: Connection failed. Is Ollama running? ({e})"

"""
This View class; handles the 'Are you sure?' confirmation for clearing memory.
input: interaction (discord.Interaction).
output: None.
"""
class ConfirmClearView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=30)

    @discord.ui.button(label="Confirm Reset", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        async with aiosqlite.connect("database.db") as db:
            await db.execute("DELETE FROM users WHERE user_id = ?", (str(interaction.user.id),))
            await db.commit()
        await interaction.response.edit_message(content="Memory wiped. We are strangers now.", view=None)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="Reset cancelled.", view=None)

class PersonalityView(discord.ui.View):
    def __init__(self, user_id):
        super().__init__(timeout=60)
        self.user_id = user_id

    async def set_personality(self, interaction, personality):
        user = await get_user(str(self.user_id))
        
        memories = json.loads(user[2]) if user else {}
        traits = user[3] if user else ""
        mood = user[4] if user else "Neutral"

        if personality not in memories:
            memories[personality] = ""

        await save_user(str(self.user_id), personality, memories, traits, mood)
        await interaction.response.edit_message(
            content=f"Personality set to **{personality.capitalize()}**.",
            view=None
        )

    @discord.ui.button(label="Gym Bro", style=discord.ButtonStyle.primary)
    async def gym(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.set_personality(interaction, "gym")

    @discord.ui.button(label="Shy Support", style=discord.ButtonStyle.success)
    async def shy(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.set_personality(interaction, "shy")

    @discord.ui.button(label="Dominant", style=discord.ButtonStyle.danger)
    async def dominant(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.set_personality(interaction, "dominant")

    @discord.ui.button(label="Bobu", style=discord.ButtonStyle.secondary)
    async def bobu(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.set_personality(interaction, "bobu")

@tree.command(name="start-axis", description="Initialize your AI companion")
async def start_axis(interaction: discord.Interaction):
    user = await get_user(str(interaction.user.id))
    if user:
        await interaction.response.send_message("You already have memories! Use `/clear-axis` to restart.", ephemeral=True)
        return
    await interaction.response.send_message("Who would you like to talk to?", view=PersonalityView(interaction.user.id), ephemeral=True)

@tree.command(name="clear-axis", description="Permanently delete your memory")
async def clear_axis(interaction: discord.Interaction):
    await interaction.response.send_message("Are you sure?", view=ConfirmClearView(), ephemeral=True)

"""
This function; handles chat logic and uses a semaphore to prevent concurrent AI crashes.
input: interaction (discord.Interaction), prompt (string), private (bool).
output: None.
"""
async def handle_chat(interaction, prompt, private=False):
    await interaction.response.defer(ephemeral=private)
    user = await get_user(str(interaction.user.id))

    if not user:
        await interaction.followup.send("Please use `/start-axis` first!", ephemeral=True)
        return

    personality = user[1]
    memories = json.loads(user[2])
    traits = user[3] or "None"
    mood = user[4] or "Neutral"
    history = memories.get(personality, "")

    full_prompt = f"Instruction: {PERSONALITIES[personality]}\nTraits: {traits}\nMood: {mood}\nHistory: {history}\nUser: {prompt}\nAssistant:"

    # The Semaphore ensures only one user hits the AI at a time. 
    # Others will wait here until the first one is done.
    async with ai_semaphore:
        response = await ask_ai(full_prompt)

    # Memory Management
    new_history = history + f"\nUser: {prompt}\nAssistant: {response}"
    lines = new_history.split("\n")
    memories[personality] = "\n".join(lines[-MEMORY_LIMIT:])

    await save_user(str(interaction.user.id), personality, memories, traits, mood)
    await interaction.followup.send(response)

@tree.command(name="say", description="Talk to Axis publicly")
async def say(interaction: discord.Interaction, prompt: str):
    await handle_chat(interaction, prompt, private=False)

@tree.command(name="whisper", description="Talk to Axis in DMs")
async def whisper(interaction: discord.Interaction, prompt: str):
    await handle_chat(interaction, prompt, private=True)

@client.event
async def on_ready():
    await init_db()
    await tree.sync()
    print(f"Axis is online as {client.user}")

client.run(TOKEN)