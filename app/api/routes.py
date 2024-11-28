from fastapi import APIRouter
from app.models.request_models import SynonymRequest
from app.services.synonym_service import generate_and_rank_synonyms

router = APIRouter()

@router.post("/generate-synonyms/")
async def generate_synonyms_api(request: SynonymRequest):
    return generate_and_rank_synonyms(request.word, request.context, request.contextual_only)

# async def generate_synonyms_api(request: SynonymRequest):
#     return generate_synonyms(request.word)