"""
Synonym Service Module

This module provides functionality for generating and ranking synonyms using OpenAI's GPT and embedding models.
It includes three main functions:

- generate_synonyms: Generates synonyms using GPT model
- rank_synonyms: Ranks synonyms using embedding-based semantic similarity 
- generate_and_rank_synonyms: Combines generation and ranking into one workflow

The service can optionally consider contextual information when generating and ranking synonyms.
"""

import openai
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings

# Configure OpenAI API key
openai.api_key = settings.openai_api_key

def generate_synonyms(word: str, context: str = None) -> dict:
    """
    Generates a list of 10 synonyms for the given word using GPT-4o-mini,
    optionally considering a context, and returns the response in JSON format.
    
    Args:
        word (str): The input word to generate synonyms for.
        context (str, optional): A sentence or description providing context for the word.
        
    Returns:
        dict: A dictionary containing:
            - word (str): The original input word
            - synonyms (list): List of generated synonym strings
            
    Raises:
        ValueError: If the API response cannot be parsed as valid JSON
    """
    if context:
        prompt = (
            f"Generate 10 diverse synonyms for the word '{word}' by considering different domains like literature, finance, science, casual conversation, and technical jargon. "
            f"in the context of this phrase:'{context}'. Return the output as a JSON object without any extra text or explanation in this format: "
            f"{{'word': '{word}', 'synonyms': ['synonym1', 'synonym2', ...]}}"
        )
    else:
        prompt = (        
            f"Generate 10 diverse synonyms for the word '{word}' by considering different domains like literature, finance, science, casual conversation, and technical jargon. "
            f"Return the output as a JSON object without any extra text or explanation in this format: "
            f"{{'word': '{word}', 'synonyms': ['synonym1', 'synonym2', ...]}}"
        )
    
    # Call GPT API to generate synonyms
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates diverse synonyms by considering various domains, contexts, and usages for a word."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.8,
    )

    print(response)

    # Extract and clean up the content
    content = response['choices'][0]['message']['content'].strip()
    
    # Remove code block markers if present
    if content.startswith("```json"):
        content = content[7:]  # Remove leading ```json
    if content.endswith("```"):
        content = content[:-3]  # Remove trailing ```
    
    import json
    # Parse JSON response
    try:
        synonyms_data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}\nResponse Content: {content}")
    
    return synonyms_data

def rank_synonyms(word: str, synonyms: list, context: str = None) -> list:
    """
    Ranks synonyms based on their semantic similarity to the input word and optional context.
    Uses OpenAI's text-embedding-3-small model to compute embeddings and cosine similarity for ranking.
    
    Args:
        word (str): The input word to compare against.
        synonyms (list): List of synonym strings to rank.
        context (str, optional): A sentence or description providing additional context for ranking.
        
    Returns:
        list: Sorted list of (synonym, similarity_score) tuples, ordered by descending similarity.
             The similarity scores are between 0 and 1, where 1 indicates highest similarity.
    """
    # Combine input word and synonyms for embedding
    all_words = [word] + synonyms
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=all_words
    )
    
    # Extract vector embeddings from response
    embeddings = [res['embedding'] for res in response['data']]
    
    # Compute similarity between input word and each synonym
    input_word_embedding = [embeddings[0]]  # Embedding for input word
    synonym_embeddings = embeddings[1:]  # Embeddings for all synonyms
    similarity_scores = cosine_similarity(input_word_embedding, synonym_embeddings).flatten()
    
    # Adjust scores based on context if provided
    if context:
        # Get embedding for the context
        context_response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=[context]
        )
        context_embedding = context_response['data'][0]['embedding']
        
        # Compute similarity between context and each synonym
        context_similarity_scores = cosine_similarity([context_embedding], synonym_embeddings).flatten()
        
        # Combine scores with a weighted average (e.g., 50% weight for each)
        adjusted_similarity_scores = [
            0.1 * s + 0.9 * cs for s, cs in zip(similarity_scores, context_similarity_scores)
        ]
    else:
        adjusted_similarity_scores = similarity_scores
    
    # Rank synonyms by combined similarity scores
    ranked_synonyms = sorted(
        zip(synonyms, adjusted_similarity_scores),
        key=lambda x: x[1],  # Sort by similarity score
        reverse=True  # Highest score first
    )
    
    return ranked_synonyms

def generate_and_rank_synonyms(word: str, context: str = None, contextual_only: bool = False) -> dict:
    """
    Generates and ranks synonyms for a given word using GPT and embeddings,
    optionally considering a provided context.
    
    This function combines the generation and ranking steps into a single workflow.
    If contextual_only is True, both generation and ranking will consider the context.
    
    Args:
        word (str): The input word to generate and rank synonyms for.
        context (str, optional): A sentence or description providing additional context.
        contextual_only (bool): If True, only generates contextual synonyms.
                              If False, generates general synonyms but still ranks with context.
        
    Returns:
        dict: Dictionary containing:
            - word (str): The original input word
            - synonyms (list): Unranked list of generated synonyms
            - ranked_synonyms (list): Same synonyms sorted by relevance to word/context
    """
    # Step 1: Generate synonyms using GPT with optional context
    if contextual_only:
        synonyms_data = generate_synonyms(word, context)
    else:
        synonyms_data = generate_synonyms(word, None)
    
    # Step 2: Rank synonyms by semantic similarity with optional contextual adjustment
    ranked_synonyms = rank_synonyms(word, synonyms_data["synonyms"], context)
    
    return {
        "word": word,
        "synonyms":  [syn for syn in synonyms_data["synonyms"]],
        "ranked synonyms": [syn for syn, _  in ranked_synonyms]
    }