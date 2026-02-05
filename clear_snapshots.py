# clear_snapshots.py
import asyncio
import aiosqlite

async def clear_all_snapshots():
    async with aiosqlite.connect("octivault_trader.db") as db:
        await db.execute("DELETE FROM symbol_snapshots")
        await db.commit()
        print("✅ All snapshot rows deleted.")

asyncio.run(clear_all_snapshots())
