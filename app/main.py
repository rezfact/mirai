from fastapi import FastAPI
from app.routes import stock
from app.utils.db_helper import init_db

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    init_db()

app.include_router(stock.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)