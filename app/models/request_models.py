from pydantic import BaseModel

class SynonymRequest(BaseModel):
    word: str
    context: str = None
    contextual_only: bool = False