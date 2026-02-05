# filename: purge_snapshot.py
import asyncio
import aiosqlite

async def purge_snapshots():
    async with aiosqlite.connect("octivault_trader.db") as db:
        await db.execute("DELETE FROM symbol_snapshots")
        await db.commit()
        print("✅ Purged old snapshots.")

asyncio.run(purge_snapshots())
