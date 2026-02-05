# drop_and_recreate_snapshot_table.py
import asyncio
import aiosqlite

async def reset_table():
    async with aiosqlite.connect("octivault_trader.db") as db:
        await db.execute("DROP TABLE IF EXISTS symbol_snapshots")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS symbol_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
        print("✅ symbol_snapshots table recreated with correct schema.")

asyncio.run(reset_table())
