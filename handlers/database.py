import json
import logging
import aiosqlite

from config import DB_PATH

log = logging.getLogger("Axis")


class Database:
    """Async SQLite wrapper. Call ``await db.init()`` once before first use."""

    def __init__(self, path: str = DB_PATH) -> None:
        self.path = path

    # ─── Schema ───────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Create the users table and migrate any missing columns."""
        async with aiosqlite.connect(self.path) as db:
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
            # Migrate columns that may be absent on older database versions.
            for col, definition in [
                ("user_name",    "TEXT NOT NULL DEFAULT ''"),
                ("prompt_count", "INTEGER NOT NULL DEFAULT 0"),
                ("gender",       "TEXT NOT NULL DEFAULT 'unspecified'"),
            ]:
                try:
                    await db.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
                except Exception:
                    pass  # Column already exists; safe to ignore.
            await db.commit()
        log.info("Database initialised.")

    # ─── CRUD ─────────────────────────────────────────────────────────────────

    async def get_user(self, user_id: str) -> tuple | None:
        """
        Fetch a single user row.
        Returns: (user_id, personality, memories_json, traits, mood,
                  user_name, prompt_count, gender) or None.
        """
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            )
            return await cursor.fetchone()

    async def save_user(
        self,
        user_id: str,
        personality: str,
        memories: dict,
        traits: str,
        mood: str,
        user_name: str = "",
        prompt_count: int = 0,
        gender: str = "unspecified",
    ) -> None:
        """Insert or replace a full user row."""
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO users "
                "(user_id, current_personality, memories, traits, mood, "
                " user_name, prompt_count, gender) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    personality,
                    json.dumps(memories),
                    traits,
                    mood,
                    user_name,
                    prompt_count,
                    gender,
                ),
            )
            await db.commit()

    async def delete_user(self, user_id: str) -> None:
        """Permanently remove a user row."""
        async with aiosqlite.connect(self.path) as db:
            await db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            await db.commit()

    async def user_exists(self, user_id: str) -> bool:
        return (await self.get_user(user_id)) is not None

    async def get_top_chatters(self, limit: int = 10) -> list[tuple]:
        """Return up to *limit* rows ordered by prompt_count descending."""
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT user_id, current_personality, prompt_count "
                "FROM users ORDER BY prompt_count DESC LIMIT ?",
                (limit,),
            )
            return await cursor.fetchall()

    # ─── Row helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def row_get(user: tuple, index: int, default: str = "") -> str:
        """Safely read any column from a user tuple without IndexError."""
        if len(user) > index and user[index]:
            return user[index]
        return default

    @staticmethod
    def row_gender(user: tuple) -> str:
        """Safely read the gender column (index 7)."""
        return Database.row_get(user, 7, "unspecified")

    # ─── Memory helpers ───────────────────────────────────────────────────────

    @staticmethod
    def parse_memories(raw: str | None) -> dict:
        """Deserialise the memories JSON string; returns {} on any failure."""
        if not raw:
            return {}
        try:
            result = json.loads(raw)
            return result if isinstance(result, dict) else {}
        except (json.JSONDecodeError, TypeError):
            log.warning("Failed to parse memories JSON; resetting to empty.")
            return {}

    @staticmethod
    def trim_history(history: str, limit: int) -> str:
        """Keep only the most recent *limit* exchanges (1 exchange = 2 lines)."""
        lines = [line for line in history.split("\n") if line.strip()]
        return "\n".join(lines[-(limit * 2):])
