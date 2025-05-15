from fastapi import APIRouter

router = APIRouter()

@router.get("/hello")
def hello():
    return {"msg": "Hello from FastAPI"}
