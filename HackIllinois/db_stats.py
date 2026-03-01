import sys

import asyncio

from cortex import AsyncCortexClient

SERVER = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"

COLLECTION = "News_Articles"


async def main():
    async with (
        AsyncCortexClient(SERVER) as client
    ):  # with makes sure this closes properly at the end of the with condition
        version, _ = await client.health_check()
        print(f"\n✓ Connected to {version}")
        count = await client.count(COLLECTION)
        print(f"\n3. Vector count: {count}")

        print("\n6. Async scroll...")
        records, next_off = await client.scroll(COLLECTION, limit=20)
        print(f"   Scrolled {len(records)} records")

        stats = await client.get_stats(COLLECTION)
        if stats:
            print(f"\n7. Collection stats: {stats.total_vectors} vectors")


if __name__ == "__main__":
    asyncio.run(main())
