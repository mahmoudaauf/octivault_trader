# inspect_snapshot_content.py
import asyncio
import aiosqlite
import json

async def check_snapshot():
    async with aiosqlite.connect("octivault_trader.db") as db:
        async with db.execute("SELECT symbols, created_at FROM symbol_snapshots ORDER BY created_at DESC LIMIT 1") as cursor:
            row = await cursor.fetchone()
            if row:
                symbols_json, created_at = row
                print(f"📦 Created At: {created_at}")
                print(f"📦 Raw JSON: {symbols_json[:300]}...")  # Trim long output
                try:
                    parsed = json.loads(symbols_json)
                    print(f"✅ Parsed Type: {type(parsed)}")
                    if isinstance(parsed, list):
                        print(f"✅ Looks good. Snapshot is a list of {len(parsed)} symbol dicts.")
                    else:
                        print("❌ Snapshot is NOT a list.")
                except Exception as e:
                    print(f"❌ JSON load failed: {e}")
            else:
                print("❌ No snapshot found.")

asyncio.run(check_snapshot())
