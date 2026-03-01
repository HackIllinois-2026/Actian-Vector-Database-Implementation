from fastapi import FastAPI

from cortex import AsyncCortexClient

SERVER = "localhost:50051"
COLLECTION = "News_Articles"
app = FastAPI()

async def get_vector_count():
    async with AsyncCortexClient(SERVER) as client:
        count = await client.count(COLLECTION)
        return count

@app.get("/vector_count")
async def vector_count():
    return {"count": await get_vector_count()}