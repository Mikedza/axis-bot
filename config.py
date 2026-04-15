import os

# ─── Discord ──────────────────────────────────────────────────────────────────
# Put your Discord bot token here, or set the DISCORD_TOKEN environment variable.
TOKEN: str = os.getenv(
    "DISCORD_TOKEN",
    "token",
)

# ─── Database ─────────────────────────────────────────────────────────────────
DB_PATH: str = "database.db"

# ─── Ollama (local LLM) ───────────────────────────────────────────────────────
OLLAMA_URL: str   = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "mistral-nemo"

# ─── Image Generation (AUTOMATIC1111 Stable Diffusion WebUI) ─────────────────
# Install:  https://github.com/AUTOMATIC1111/stable-diffusion-webui
# Launch with --api flag; listens on localhost:7860 by default.
# Set IMAGE_GEN_ENABLED = True once the WebUI is running and tested.
IMAGE_GEN_ENABLED: bool = True
IMAGE_GEN_URL: str      = "http://localhost:7860/sdapi/v1/txt2img"

# ─── Behaviour ────────────────────────────────────────────────────────────────
MEMORY_LIMIT: int    = 12    # Max conversation exchanges kept per personality per user.
MAX_CONCURRENT: int  = 2     # Max simultaneous Ollama requests. Raise with caution.
REQUEST_TIMEOUT: int = 120   # Seconds before an Ollama call times out.
DISCORD_MAX_LEN: int = 1990  # Character cap per Discord message (hard limit is 2000).
