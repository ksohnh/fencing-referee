from pydantic import BaseModel
from typing import List, Dict

class TouchEvent(BaseModel):
    timestamp: float        # seconds into video
    fencer: str             # "red" or "green"
    right_of_way: bool      # True if awarded

class AnnotationBundle(BaseModel):
    attacks: List[TouchEvent] = []
    parries: List[TouchEvent] = []
    ripostes: List[TouchEvent] = []
    # add other event types as needed
