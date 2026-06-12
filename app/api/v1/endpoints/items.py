from fastapi import APIRouter, HTTPException, status
from typing import List
from app.schemas.items import ItemCreate, Item

router = APIRouter()

# অস্থায়ী in-memory ডাটাবেস
_db: List[Item] = []
_next_id = 1


@router.post("/", response_model=Item, status_code=status.HTTP_201_CREATED)
def create_item(payload: ItemCreate):
    global _next_id
    item = Item(id=_next_id, **payload.dict())
    _db.append(item)
    _next_id += 1
    return item


@router.get("/", response_model=List[Item])
def list_items():
    return _db


@router.get("/{item_id}", response_model=Item)
def get_item(item_id: int):
    for it in _db:
        if it.id == item_id:
            return it
    raise HTTPException(status_code=404, detail="Item not found")