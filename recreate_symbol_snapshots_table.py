# recreate_symbol_snapshots_table.py
import asyncio
import aiosqlite

async def recreate_table():
    async with aiosqlite.connect("octivault_trader.db") as db:
        await db.execute("DROP TABLE IF EXISTS symbol_snapshots")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS symbol_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbols TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
        print("✅ Recreated symbol_snapshots with column 'symbols'.")

asyncio.run(recreate_table())
