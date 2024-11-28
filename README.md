# Synonym Generation API: Advanced NLP Solution

## Project Overview

The Synonym Generator API is designed to generate and rank synonyms for a given word using OpenAI's GPT models. It supports optional context input to tailor synonyms to specific scenarios. The API provides structured JSON output for easy integration and parsing.

## Features

- **Synonym Generation**: Generates a list of synonyms for a specified word.
- **Contextual Synonyms**: Optionally generates synonyms solely based on the provided context.
- **Ranking**: Ranks synonyms based on semantic similarity to the input word and context (if provided).
- **JSON Output**: Returns results in a structured JSON format for easy parsing.

### API Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `word` | String | Input word for synonym generation | Required |
| `context` | String | Optional contextual information | Optional |
| `contextual_only` | Boolean | Strict contextual synonym generation | Optional: default = `False` |


## Example Use Case

#### Example 1 - Basic Synonym Generation from given word (**Without Context**)

 **Request:**
 ```json
     {
     "word": "cold"
     }
 ```

 **Response:**
 ```json
  {
    "word": "cold",
    "synonyms": [
        "chilly",
        "frigid",
        "unfriendly",
        "detached",
        "cool",
        "icy",
        "nonchalant",
        "frosty",
        "standoffish",
        "subzero"
    ],
    "ranked synonyms": [
        "chilly",
        "icy",
        "frigid",
        "cool",
        "frosty",
        "nonchalant",
        "unfriendly",
        "subzero",
        "standoffish",
        "detached"
       ]
   }
```

#### Example 2 - Synonym Generation from given word, with Context-Aware Synonym Ranking

 **Request:**
 ```json
   {
     "word": "cold",
     "context": "His tone was cold during the argument.",
     "contextual_only": false
   }
 ```

 **Response:**
 ```json
  {
    "word": "cold",
    "synonyms": [
        "chilly",
        "frigid",
        "unfriendly",
        "detached",
        "cool",
        "icy",
        "nonchalant",
        "frosty",
        "standoffish",
        "subzero"
    ],
     "ranked synonyms": [
       "detached",
       "unfriendly",
       "standoffish",
       "frosty",
       "icy",
       "chilly",
       "frigid",
       "nonchalant",
       "cool",
       "subzero"
     ]
   }
```

#### Example 3 - Synonym Generation Using Word and Context, with Contextual Ranking

 **Request:**
 ```json
   {
     "word": "cold",
     "context": "His tone was cold during the argument.",
     "contextual_only": true
   }
 ```

 **Response:**
 ```json
  {
    "word": "cold",
    "synonyms": [
        "unemotional",
        "distant",
        "indifferent",
        "aloof",
        "dispassionate",
        "frosty",
        "icy",
        "frigid",
        "detached",
        "clinical"
    ],
    "ranked synonyms": [
        "frigid",
        "dispassionate",
        "frosty",
        "unemotional",
        "icy",
        "indifferent",
        "aloof",
        "distant",
        "detached",
        "clinical"
    ]
   }
```

## How Synonyms are Generated

The synonym generation process is designed to provide diverse and meaningful options, tailored to the input word and optional context. OpenAI's GPT models are leveraged to ensure linguistic accuracy and relevance across multiple domains.

### Generation Process

1. **Word-Based Generation (No Context)**  
   - Synonyms are generated by analyzing all possible meanings of the word.  
   - This ensures a comprehensive list that spans different usages and interpretations.  
   - **Example:** For "bank," synonyms include "financial institution," "repository," and "riverbank," covering financial, geographical, and general meanings.

2. **Strict Context-Aware Generation (Context Provided and `contextual_only = true`)**  
   - The model incorporates the given context in the prompt to exclusively generate synonyms relevant to the specified domain.  
   - This approach filters out all meanings unrelated to the context.  
   - **Example:** For "bank" with the context "river edge," only synonyms like "shore" and "riverbank" are generated, avoiding financial or irrelevant terms.

### Generation Logic

- **Model**: *GPT-4o-mini* (GPT-4 family's most affordable model)
- **Prompt Design**:  The generation prompt is strategically crafted to:
    - Request synonyms from diverse domains (e.g., finance, geography, etc.)
    - Consider multiple usage contexts (check if a specific context is provided, and if so, does the user want to generate synonyms ONLY relevant to that context)
    - Ensure structured JSON output (e.g., "synonyms": ["synonym1", "synonym2", "synonym3"])
- **Temperature Control**: Set to 0.8 for a balance of creativity and consistency.  


## How Synonyms are Ranked

Synonym ranking ensures the most relevant options are presented first, enhancing usability and reducing ambiguity. The approach dynamically adjusts based on the presence and importance of context, using **text-embedding-3-small** to compute semantic similarity between the input word and its synonyms.

### Ranking Scenarios

1. **Word-Centric Ranking (No Context)**  
   - Synonyms are ranked solely by their semantic closeness to the word.  
   - This approach highlights general meanings, ensuring the most common interpretations appear first.  
   - **Example:** For "bank," synonyms like "financial institution" and "riverbank" rank highest due to their close association with the word's general usage.

2. **Context-Enhanced Ranking (Context Provided)**  
   - Synonyms matching the specified context are weighted more heavily.  
   - Other synonyms are included but deprioritised in the ranking.  
   - **Example:** For "bank" with the context "monetary institution," terms like "currency exchange" and "savings house" rank higher, while unrelated meanings such as "slope" are ranked lower.

3. **Strict Contextual Ranking (`contextual_only = true`)**  
   - Only synonyms directly relevant to the context are included.  
   - These synonyms are further ranked by their semantic alignment with the context.  
   - **Example:** For "bank" with the context "river edge," synonyms like "riverbank," "shore," and "embankment" are included, while financial terms are excluded entirely.

### Ranking Algorithm

- **Embedding Model**: *text-embedding-3-small*
- **Process**:  
  1. Computes cosine similarity between embeddings of the input word and generated synonyms.  
  2. Dynamically adjusts scores based on context, giving more weight to synonyms matching the specified context and prioritising domain-specific relevance when provided.  
  3. Filters out unrelated synonyms entirely when `contextual_only` is enabled.  


## Technical Architecture

### Technical Stack

- **Web Framework**: FastAPI
- **AI Models**: OpenAI (GPT-4o-mini, text-embedding-3-small)
- **Embedding Computation**: scikit-learn
- **Configuration Management**: Pydantic
- **Deployment**: uvicorn

### Prerequisites

- Python 3.8 or higher (tested on 3.10)
- An OpenAI API key
- Required Python packages (listed in `requirements.txt`)

### Folder Structure

```
synonym_api/
│
├── app/
│   ├── api/
│   │   ├── routes.py           # API endpoint definitions
│   ├── core/
│   │   ├── config.py           # Configuration management
│   ├── services/
│   │   ├── synonym_service.py  # Core synonym generation and rankinglogic
│   ├── models/
│   │   ├── request_models.py   # Input validation models
│
├── .env                        # For storing OPENAI_API_KEY
├── requirements.txt
└── README.md
```

### Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/synonym_api.git
    cd synonym_api
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the OpenAI API key:**
    - Edit the `.env` file in the root directory.
    - Add your OpenAI API key:
      ```
      OPENAI_API_KEY=your_openai_api_key_here
      ```

4. **Run the application:**
    ```bash
    uvicorn app.main:app --reload
    ```

5. **Test the API:**
    - Test the API using Postman or curl
    - Send a POST request to http://127.0.0.1:8000/generate-synonyms/ with a JSON body containing the word, context (if applicable), and contextual_only (if applicable).

### API Usage Examples

Here are three example requests demonstrating different ways to use the API:

1. Basic Request - Get All Possible Synonyms:

    This request returns synonyms across all possible contexts and meanings of the word, and ranks them based on similarity to the word.

    **Request:**
    ```json
        {
        "word": "bank"
        }
    ```

    **Response:**
    ```json
        {
        "word": "bank",
        "synonyms": ["financial institution", "repository", "savings institution", "cache", "trust", "data repository", "conservatory", "riverbank", "embankment", "dike"],
        "ranked synonyms": ["financial institution", "riverbank", "savings institution", "embankment", "trust", "repository", "cache", "data repository", "dike", "conservatory"]
        }
    ```

    **Mechanism:**
    - This approach generates synonyms spanning all possible meanings of the word.
    - The ranking is based on overall similarity to the word itself, irrespective of any specific context.
    - For example, "bank" includes meanings like "financial institution" and "riverbank" ranked higher because they are more commonly associated with the word, while less-used meanings like "conservatory" or "cache" appear lower.

2. Contextual Request - Prioritise Specific Context:

    This request uses the optional context parameter to prioritise synonyms related to a specific meaning, while still potentially including other relevant synonyms. The results will be ranked with financial institution-related synonyms appearing top of the list.

    **Request:**
    ```json
        {
        "word": "bank",
        "context": "monetary institution",
        "contextual_only": false
        }
    ```

    **Response:**
    ```json
        {
        "word": "bank",
        "synonyms": ["financial institution", "repository", "slope", "reserve", "cache", "data storage", "dike", "trust", "savings house", "currency exchange"],
        "ranked synonyms": ["financial institution", "currency exchange", "reserve", "savings house", "repository", "trust", "dike", "cache", "data storage", "slope"]
        }
    ```

    **Mechanism:**
    - When a specific context is provided (e.g., "monetary institution"), synonyms related to this context are prioritised in the ranking.
    - With contextual_only filtering turned off, synonyms from other contexts are still included but ranked lower, ensuring a broad yet weighted result set.
    - For "bank" with the context of "monetary institution," terms like "financial institution," "currency exchange," and "savings house" rank highest, while non-financial terms like "slope" or "dike" are deprioritised.

3. Strict Contextual Request - Context-Only Synonyms:

    This request uses both context and contextual_only parameters to strictly generate synonyms
    synonyms matching the specified context. By setting contextual_only to true, only river
    edge-related synonyms will be returned, filtering out all other meanings, ranked with the most relevant synonyms appearing top of the list.

    **Request:**
    ```json
        {
        "word": "bank", 
        "context": "river edge",
        "contextual_only": true
        }
    ```

    **Response:**
    ```json
        {
        "word": "bank",
        "synonyms": ["shore", "embankment", "riverbank", "margin", "ledge", "coast", "fringe", "waterside", "beach", "promontory"],
        "ranked synonyms": ["riverbank", "embankment", "waterside", "beach", "shore", "coast", "margin", "promontory", "ledge", "fringe"]
        }
    ```

    **Mechanism:**
    -  When both a context and contextual_only are used, the synonyms and ranking are strictly limited to the provided context.
    - Synonyms unrelated to the context are excluded entirely.
    - For "bank" in the context of "river edge," only synonyms like "shore," "riverbank," and "waterside" are included. Financial or unrelated meanings are completely filtered out.


### 5. Error Handling and Robustness

#### Key Error Management Strategies
- Robust JSON request in prompt and response parsing
- Detailed exception handling
- API response for OpenAI integration failures

### 6. Performance Optimisation

#### Efficiency Considerations
- Minimal token usage (max 100 tokens)
- Temperature-controlled creativity (0.8)
- Efficient embedding computation
- Potential future caching mechanisms

## Future Improvements

- Implement caching mechanisms to reduce redundant OpenAI API calls.
- Enhance error handling with more specific responses.
- Explore additional NLP libraries / Open Source models for improved synonym generation accuracy.

## Contributing

Congratulations, you’ve unlocked the ‘Finished the Docs’ achievement! Either we’re doing something right, or you’re just too polite to stop. Either way, thank you!

## License

This project is licensed under the MIT License.
