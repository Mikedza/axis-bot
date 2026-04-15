import logging

import discord
from discord import app_commands
from discord.ext import commands

from handlers.music_handler import MusicHandler

log = logging.getLogger("Axis")


class MusicCog(commands.Cog):
    """Slash commands for voice-channel music playback."""

    def __init__(self, bot: commands.Bot, music: MusicHandler) -> None:
        self.bot   = bot
        self.music = music

    # ─── /play ────────────────────────────────────────────────────────────────

    @app_commands.command(
        name="play",
        description="Play a song in your voice channel (search term or URL)",
    )
    async def play(self, interaction: discord.Interaction, query: str) -> None:
        if not interaction.guild:
            await interaction.response.send_message(
                "This command only works in a server.", ephemeral=True
            )
            return

        voice_state = interaction.user.voice  # type: ignore[union-attr]
        if not voice_state or not voice_state.channel:
            await interaction.response.send_message(
                "You need to be in a voice channel first.", ephemeral=True
            )
            return

        await interaction.response.defer()

        guild_id = interaction.guild.id
        channel  = voice_state.channel

        vc = self.music.voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            vc = await channel.connect()
            self.music.voice_clients[guild_id] = vc
        elif vc.channel != channel:
            await vc.move_to(channel)

        track = await self.music.fetch_track(query, str(interaction.user))
        if not track:
            await interaction.followup.send(
                "Could not find anything for that query. Try a different search or URL."
            )
            return

        already_playing = vc.is_playing() or vc.is_paused()
        self.music.queues[guild_id].append(track)
        position = len(self.music.queues[guild_id])

        embed             = discord.Embed(color=discord.Color.green())
        embed.description = f"**{track['title']}**"
        embed.add_field(name="Duration",     value=MusicHandler.format_duration(track["duration"]), inline=True)
        embed.add_field(name="Requested by", value=track["requester"],                              inline=True)

        if already_playing:
            embed.title = "Added to Queue 🎶"
            embed.add_field(name="Position", value=f"#{position}", inline=True)
        else:
            embed.title = "Now Playing 🎵"
            await self.music.play_next(guild_id)

        await interaction.followup.send(embed=embed)
        log.info(f"[{interaction.user}] Queued: {track['title']}")

    # ─── /play-queue ──────────────────────────────────────────────────────────

    @app_commands.command(name="play-queue", description="Show the current music queue")
    async def play_queue(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message(
                "This command only works in a server.", ephemeral=True
            )
            return

        queue = self.music.queues[interaction.guild.id]
        if not queue:
            await interaction.response.send_message(
                "The queue is empty. Use `/play` to add songs.", ephemeral=True
            )
            return

        embed = discord.Embed(title="Music Queue 🎵", color=discord.Color.blurple())
        for i, track in enumerate(queue, start=1):
            label = "Now Playing 🎵" if i == 1 else f"#{i}"
            embed.add_field(
                name=f"{label} — {track['title']}",
                value=(
                    f"Duration: {MusicHandler.format_duration(track['duration'])} "
                    f"| Requested by: {track['requester']}"
                ),
                inline=False,
            )
        await interaction.response.send_message(embed=embed)

    # ─── /skip ────────────────────────────────────────────────────────────────

    @app_commands.command(name="skip", description="Skip the current song")
    async def skip(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message(
                "This command only works in a server.", ephemeral=True
            )
            return

        vc = self.music.voice_clients.get(interaction.guild.id)
        if not vc or not vc.is_playing():
            await interaction.response.send_message(
                "Nothing is playing right now.", ephemeral=True
            )
            return

        vc.stop()  # Triggers after_playing → pops track → calls play_next.
        await interaction.response.send_message("Skipped ⏭️")
        log.info(f"[{interaction.user}] Skipped track in guild {interaction.guild.id}.")

    # ─── /leave-call ──────────────────────────────────────────────────────────

    @app_commands.command(name="leave-call", description="Make Axis leave the voice channel")
    async def leave_call(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message(
                "This command only works in a server.", ephemeral=True
            )
            return

        guild_id = interaction.guild.id
        vc       = self.music.voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            await interaction.response.send_message(
                "I am not in a voice channel.", ephemeral=True
            )
            return

        self.music.queues[guild_id].clear()
        await vc.disconnect()
        self.music.voice_clients.pop(guild_id, None)
        await interaction.response.send_message("Left the voice channel and cleared the queue. 👋")
        log.info(f"[{interaction.user}] Bot disconnected from guild {guild_id}.")

    # ─── /reset-queue ─────────────────────────────────────────────────────────

    @app_commands.command(name="reset-queue", description="Clear the entire music queue")
    async def reset_queue(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message(
                "This command only works in a server.", ephemeral=True
            )
            return

        guild_id = interaction.guild.id
        vc       = self.music.voice_clients.get(guild_id)
        if vc and vc.is_playing():
            vc.stop()
        self.music.queues[guild_id].clear()
        await interaction.response.send_message("Queue cleared. 🗑️")
        log.info(f"[{interaction.user}] Queue reset in guild {guild_id}.")
