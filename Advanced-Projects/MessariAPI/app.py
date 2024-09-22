import uvicorn
from routers import router
from fastapi import FastAPI

app = FastAPI(title="MessariAPI")
app.include_router(router)

# ouverture de l'url sur un moteur de recherche
if __name__ == '__main__':
    uvicorn.run(app)
