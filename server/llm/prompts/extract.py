EXTRACT_PROMPT = """You are an expert knowledge graph extraction specialist.

Your task is to extract the most important entities and relationships from the given text for building a high-quality knowledge graph.

Return ONLY a valid JSON object in the exact format below. Do not add any explanations, markdown, or extra text.

**Output exactly this format — nothing else**:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "...", "description": "..."}}
  ]
}}
"""
