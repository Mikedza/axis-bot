import logging
import discord
from discord import app_commands
from discord.ext import commands
from handlers.music_handler import MusicHandler

log = logging.getLogger("Axis")

class MusicCog(commands.Cog):
    """Slash commands for voice-channel music playback."""

    def __init__(self, bot: commands.Bot, music: MusicHandler) -> None:
        self.bot = bot
        self.music = music

    @app_commands.command(name="play", description="Play a song (search or URL)")
    async def play(self, interaction: discord.Interaction, query: str) -> None:
        if not interaction.guild:
            return await interaction.response.send_message("Servers only.", ephemeral=True)

        voice_state = interaction.user.voice # type: ignore
        if not voice_state or not voice_state.channel:
            return await interaction.response.send_message("Join a voice channel first.", ephemeral=True)

        await interaction.response.defer()
        guild_id = interaction.guild.id
        
        # Connection Logic
        vc = self.music.voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            vc = await voice_state.channel.connect()
            self.music.voice_clients[guild_id] = vc
        elif vc.channel != voice_state.channel:
            await vc.move_to(voice_state.channel)

        track = await self.music.fetch_track(query, str(interaction.user))
        if not track:
            return await interaction.followup.send("Could not find that track.")

        already_playing = vc.is_playing() or vc.is_paused()
        self.music.queues[guild_id].append(track)
        
        embed = discord.Embed(color=discord.Color.green(), description=f"**{track['title']}**")
        embed.add_field(name="Duration", value=MusicHandler.format_duration(track["duration"]))
        embed.add_field(name="Requested by", value=track["requester"])

        if already_playing:
            embed.title = "Added to Queue 🎶"
            embed.add_field(name="Position", value=f"#{len(self.music.queues[guild_id])}")
        else:
            embed.title = "Now Playing 🎵"
            await self.music.play_next(guild_id)

        await interaction.followup.send(embed=embed)

    @app_commands.command(name="play-queue", description="Show the music queue")
    async def play_queue(self, interaction: discord.Interaction) -> None:
        queue = self.music.queues.get(interaction.guild_id, []) # type: ignore
        if not queue:
            return await interaction.response.send_message("Queue is empty.", ephemeral=True)

        embed = discord.Embed(title="Music Queue 🎵", color=discord.Color.blurple())
        for i, track in enumerate(queue[:10], start=1):
            label = "Now Playing" if i == 1 else f"#{i}"
            embed.add_field(
                name=f"{label} — {track['title']}",
                value=f"By: {track['requester']}",
                inline=False
            )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="skip", description="Skip the current song")
    async def skip(self, interaction: discord.Interaction) -> None:
        vc = self.music.voice_clients.get(interaction.guild_id) # type: ignore
        if not vc or not vc.is_playing():
            return await interaction.response.send_message("Nothing is playing.", ephemeral=True)

        vc.stop()
        await interaction.response.send_message("Skipped ⏭️")

    @app_commands.command(name="leave-call", description="Disconnect the bot")
    async def leave_call(self, interaction: discord.Interaction) -> None:
        guild_id = interaction.guild_id # type: ignore
        vc = self.music.voice_clients.get(guild_id)
        if vc:
            self.music.queues[guild_id].clear()
            await vc.disconnect()
            self.music.voice_clients.pop(guild_id, None)
            await interaction.response.send_message("Goodbye! 👋")
        else:
            await interaction.response.send_message("Not in a channel.", ephemeral=True)

    # ─── /pause ───────────────────────────────────────────────────────────────

    @app_commands.command(name="pause", description="Pause the current track")
    async def pause(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            return await interaction.response.send_message("Servers only.", ephemeral=True)

        vc = self.music.voice_clients.get(interaction.guild.id)
        
        if not vc or not vc.is_playing():
            return await interaction.response.send_message("Nothing is playing right now.", ephemeral=True)

        if vc.is_paused():
            return await interaction.response.send_message("The music is already paused.", ephemeral=True)

        vc.pause()
        await interaction.response.send_message("Paused ⏸️")
        log.info(f"[{interaction.user}] Paused playback in guild {interaction.guild.id}.")

    # ─── /resume ──────────────────────────────────────────────────────────────

    @app_commands.command(name="resume", description="Resume the paused track")
    async def resume(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            return await interaction.response.send_message("Servers only.", ephemeral=True)

        vc = self.music.voice_clients.get(interaction.guild.id)

        if not vc:
            return await interaction.response.send_message("I am not in a voice channel.", ephemeral=True)

        if not vc.is_paused():
            return await interaction.response.send_message("The music is not paused.", ephemeral=True)

        vc.resume()
        await interaction.response.send_message("Resumed ▶️")
        log.info(f"[{interaction.user}] Resumed playback in guild {interaction.guild.id}.")

async def setup(bot: commands.Bot) -> None:
    # This assumes you instantiate MusicHandler once and pass it to the Cog
    handler = MusicHandler()
    await bot.add_cog(MusicCog(bot, handler))