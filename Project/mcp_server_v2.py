import os
import sys
import faiss
import pickle
import json
import re
import numpy as np
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Load the embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load pre-built lecture index
INDEX_FILE = "lectures_v2.index"
TEXTS_FILE = "lectures_v2.pkl"

if not os.path.exists(INDEX_FILE) or not os.path.exists(TEXTS_FILE):
    print("Error: Lecture index files not found. Please run build_index_v2.py first!", file=sys.stderr)
    exit(1)

index = faiss.read_index(INDEX_FILE)
with open(TEXTS_FILE, "rb") as f:
    texts = pickle.load(f)

# Load AI configuration
AI_CONFIG_FILE = "ai_config.json"
if os.path.exists(AI_CONFIG_FILE):
    with open(AI_CONFIG_FILE, "r") as f:
        ai_config = json.load(f)
else:
    ai_config = {"examples": [], "dataset_sources": []}

grounding_policy = ai_config.get("grounding_policy", {})
NOT_FOUND_RESPONSE = grounding_policy.get("not_found_response", "Not found in provided materials")

# Load datasets into memory at startup
print("Loading datasets...", file=sys.stderr)
DATASETS = {}

def load_local_jsonl(file_path):
    """Load a local JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    class LocalDataset:
        def __init__(self, data):
            self.data = data
            self.column_names = list(data[0].keys()) if data else []
        def __len__(self):
            return len(self.data)
        def __iter__(self):
            return iter(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    return LocalDataset(data)

for ds_config in ai_config.get("dataset_sources", []):
    ds_name = ds_config["name"]
    try:
        print(f"  Loading {ds_name}...", file=sys.stderr)
        
        if ds_config.get("type") == "local" or ds_name.startswith("local:"):
            file_path = ds_config.get("file_path", ds_name.replace("local:", ""))
            dataset = load_local_jsonl(file_path)
        else:
            dataset = load_dataset(ds_name, split=ds_config.get("split", "train"))
        
        DATASETS[ds_name] = {
            "data": dataset,
            "field_overrides": ds_config.get("field_overrides", {}),
            "name": ds_name
        }
        print(f"  ✓ Loaded {len(dataset)} examples from {ds_name}", file=sys.stderr)
    except Exception as e:
        print(f"  ✗ Failed to load {ds_name}: {e}", file=sys.stderr)

print(f"✓ Loaded {len(DATASETS)} datasets", file=sys.stderr)

# Create MCP server
mcp = FastMCP("lecture-assistant-v5-ai-enhanced")

def call_local_llm(prompt, max_tokens=200, temperature=0.1):
    """Call local Ollama LLM"""
    try:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b-instruct",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9
                }
            },
            timeout=240
        )
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            print(f"LLM: {result[:80]}...", file=sys.stderr)
            return result
    except Exception as e:
        print(f"❌ LLM Error: {e}", file=sys.stderr)
        print("⚠️  Make sure Ollama is running: ollama serve", file=sys.stderr)
    return None

def get_dataset_field(row, field_overrides, field_type):
    """Extract field from dataset row"""
    field_name = field_overrides.get(field_type)
    if isinstance(field_name, list):
        field_name = field_name[0] if field_name else None
    if field_name and field_name in row:
        return row[field_name]
    
    fallbacks = {
        "question": ["question", "Question", "query", "prompt", "Problem"],
        "answer": ["answer", "Answer", "solution", "ground_truth", "Rationale"],
    }
    for fallback in fallbacks.get(field_type, []):
        if fallback in row:
            return row[fallback]
    return None

def ai_enhance_query(query):
    """Use AI to expand/clarify the query for better search"""
    prompt = f"""You are a query enhancement system. Given a user's question, generate 3-5 related search terms or alternative phrasings that would help find relevant information.

User Question: {query}

Output ONLY the search terms, one per line, no explanations:"""
    
    response = call_local_llm(prompt, max_tokens=100, temperature=0.3)
    if response:
        # Parse search terms
        terms = [line.strip() for line in response.split('\n') if line.strip()]
        print(f"🤖 AI expanded query to: {terms[:3]}", file=sys.stderr)
        return terms[:3]
    return []

def ai_rerank_results(query, results, top_k=5):
    """Use AI to rerank results based on relevance"""
    if not results or len(results) <= top_k:
        return results
    
    # Build prompt with result summaries
    summaries = []
    for i, r in enumerate(results[:10], 1):
        content = r.get('content', r.get('answer', ''))[:200]
        summaries.append(f"[{i}] {content}")
    
    prompt = f"""Rank these search results by relevance to the question. Output ONLY the numbers in order (most relevant first), comma-separated.

Question: {query}

Results:
{chr(10).join(summaries)}

Ranking (e.g., "3,1,5,2,4"):"""
    
    response = call_local_llm(prompt, max_tokens=50, temperature=0.1)
    if response:
        try:
            # Parse ranking
            ranking = [int(x.strip())-1 for x in response.split(',') if x.strip().isdigit()]
            reranked = [results[i] for i in ranking if i < len(results)]
            # Add any missed results
            for r in results:
                if r not in reranked:
                    reranked.append(r)
            print(f"🤖 AI reranked results: {ranking[:5]}", file=sys.stderr)
            return reranked[:top_k]
        except:
            pass
    
    return results[:top_k]

def search_datasets(query, k=10, expanded_terms=None):
    """Search datasets with AI-enhanced terms"""
    all_results = []
    
    # Use original query + expanded terms
    search_terms = [query.lower()]
    if expanded_terms:
        search_terms.extend([t.lower() for t in expanded_terms])
    
    for ds_name, ds_info in DATASETS.items():
        dataset = ds_info["data"]
        field_overrides = ds_info["field_overrides"]
        
        for idx, row in enumerate(dataset):
            question = get_dataset_field(row, field_overrides, "question")
            answer = get_dataset_field(row, field_overrides, "answer")
            
            if not question or not answer:
                continue
            
            question_str = str(question).lower()
            answer_str = str(answer).lower()
            
            # Score based on all search terms
            score = 0
            for term in search_terms:
                if term in question_str:
                    score += 15
                if term in answer_str:
                    score += 5
                # Word overlap
                term_words = set(term.split())
                q_words = set(question_str.split())
                a_words = set(answer_str.split())
                score += len(term_words & q_words) * 2
                score += len(term_words & a_words)
            
            if score > 3:
                all_results.append({
                    "source": "dataset",
                    "dataset_name": ds_name,
                    "question": str(question)[:300],
                    "answer": str(answer)[:600],
                    "score": score,
                    "index": idx
                })
    
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:k]

def extract_answer_with_ai(query, lecture_results, dataset_results):
    """OLLAMA-ONLY extraction - No fallback"""
    
    # Build rich context
    context_parts = []
    
    if lecture_results:
        context_parts.append("=== LECTURE MATERIALS ===")
        for i, r in enumerate(lecture_results[:5], 1):
            context_parts.append(f"\n[Lecture {i}] {r['file']}, Page {r['page']}")
            context_parts.append(f"Relevance: {1 - r['distance']:.2f}")
            context_parts.append(f"Content: {r['content'][:600]}")
    
    if dataset_results:
        context_parts.append("\n\n=== CURATED Q&A EXAMPLES ===")
        for i, r in enumerate(dataset_results[:5], 1):
            context_parts.append(f"\n[Example {i}] {r['dataset_name']} (Score: {r['score']})")
            context_parts.append(f"Q: {r['question'][:200]}")
            context_parts.append(f"A: {r['answer'][:400]}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""You are a precise educational assistant. Using ONLY the materials provided below, answer the student's question clearly and accurately.

IMPORTANT RULES:
1. Base your answer ONLY on the materials below
2. Synthesize information from multiple sources when relevant
3. Be clear and educational in tone
4. If materials don't fully answer the question, say what you CAN answer and note what's missing
5. Keep answer to 2-4 sentences unless more detail is clearly needed
6. Remove any PDF formatting artifacts (bullets, strange characters)

STUDENT QUESTION: {query}

AVAILABLE MATERIALS:
{context}

ANSWER:"""

    response = call_local_llm(prompt, max_tokens=300, temperature=0.2)
    
    if not response:
        return {
            "answer": "ERROR: Ollama is not running or not responding. Please start Ollama with: ollama serve",
            "type": "error",
            "sources": {"lectures": 0, "datasets": 0}
        }
    
    # Clean up response
    cleaned = response.strip().strip('"\'')
    
    # Remove common prefixes
    prefixes = ["answer:", "the answer is:", "based on", "according to"]
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    if len(cleaned) < 10:
        return {
            "answer": NOT_FOUND_RESPONSE,
            "type": "not_found",
            "sources": {"lectures": len(lecture_results), "datasets": len(dataset_results)}
        }
    
    return {
        "answer": cleaned,
        "type": "ai_synthesis",
        "sources": {
            "lectures": len(lecture_results),
            "datasets": len(dataset_results)
        }
    }

def classify_query(query):
    """Use AI to classify query type"""
    prompt = f"""Classify this question into ONE category. Output ONLY the category name.

Categories: definition, comparison, procedure, calculation, person, time, location, general

Question: {query}

Category:"""
    
    response = call_local_llm(prompt, max_tokens=10, temperature=0.1)
    if response:
        category = response.strip().lower()
        print(f"🤖 AI classified as: {category}", file=sys.stderr)
        return category
    return 'general'

def calculate_confidence(query, answer, lecture_results, dataset_results):
    """Calculate confidence score"""
    if answer == NOT_FOUND_RESPONSE or "ERROR:" in answer:
        return 0.0
    
    score = 0.0
    
    if lecture_results:
        best_distance = lecture_results[0].get('distance', 1.0)
        score += max(0, (2 - best_distance) / 2) * 0.4
    
    if dataset_results:
        best_score = dataset_results[0].get('score', 0)
        score += min(best_score / 20, 1.0) * 0.3
    
    answer_words = len(answer.split())
    if 10 <= answer_words <= 150:
        score += 0.3
    elif 5 <= answer_words < 10:
        score += 0.15
    
    return round(min(score, 1.0), 2)

@mcp.tool()
def search_all_sources(query: str, k_lectures: int = 10, k_datasets: int = 10, use_ai_enhancement: bool = True) -> str:
    """
    AI-enhanced search across lectures and datasets.
    
    Args:
        query: The question
        k_lectures: Number of lecture results
        k_datasets: Number of dataset results  
        use_ai_enhancement: Use AI for query expansion and reranking
    
    Returns:
        JSON with AI-synthesized answer
    """
    print(f"\n🔍 Processing query: {query}", file=sys.stderr)
    
    # AI Enhancement: Expand query
    expanded_terms = []
    if use_ai_enhancement:
        expanded_terms = ai_enhance_query(query)
    
    # AI Enhancement: Classify query
    query_type = classify_query(query) if use_ai_enhancement else 'general'
    
    # Search lectures
    query_emb = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, k_lectures)
    
    lecture_results = []
    for idx, dist in zip(indices[0], distances[0]):
        if len(texts[idx]) == 5:
            file, page_num, chunk_idx, enriched_text, original_text = texts[idx]
        elif len(texts[idx]) == 4:
            file, page_num, chunk_idx, text = texts[idx]
            original_text = text
        else:
            file, page_num, text = texts[idx]
            chunk_idx = 0
            original_text = text
        
        lecture_results.append({
            "source": "lecture",
            "file": file,
            "page": page_num,
            "distance": float(dist),
            "content": original_text
        })
    
    # Search datasets with expanded terms
    dataset_results = search_datasets(query, k=k_datasets, expanded_terms=expanded_terms)
    
    # AI Enhancement: Rerank all results
    if use_ai_enhancement and (lecture_results or dataset_results):
        all_results = lecture_results + dataset_results
        all_results = ai_rerank_results(query, all_results, top_k=k_lectures + k_datasets)
        # Split back
        lecture_results = [r for r in all_results if r.get('source') == 'lecture'][:k_lectures]
        dataset_results = [r for r in all_results if r.get('source') == 'dataset'][:k_datasets]
    
    print(f"📚 Found {len(lecture_results)} lectures, {len(dataset_results)} datasets", file=sys.stderr)
    
    # OLLAMA-ONLY extraction
    extracted = extract_answer_with_ai(query, lecture_results, dataset_results)
    
    confidence = calculate_confidence(query, extracted['answer'], lecture_results, dataset_results)
    extracted['confidence'] = confidence
    
    response = {
        "query_type": query_type,
        "lecture_results": lecture_results,
        "dataset_results": dataset_results,
        "extracted_answer": extracted,
        "ai_enhancements": {
            "expanded_terms": expanded_terms,
            "reranked": use_ai_enhancement
        }
    }
    
    return json.dumps(response, indent=2)

@mcp.tool()
def ask_question(question: str) -> str:
    """Ask a question with full AI enhancement."""
    result = json.loads(search_all_sources(question, k_lectures=10, k_datasets=10, use_ai_enhancement=True))
    
    extracted = result.get("extracted_answer", {})
    answer = extracted.get("answer", NOT_FOUND_RESPONSE)
    
    sources_list = []
    for r in result["lecture_results"][:3]:
        sources_list.append(f"📄 {r['file']} (page {r['page']})")
    for r in result["dataset_results"][:2]:
        if r['score'] > 8:
            sources_list.append(f"📊 {r['dataset_name']}")
    
    return json.dumps({
        "answer": answer,
        "confidence": extracted.get("confidence", 0.0),
        "extraction_method": extracted.get("type", "unknown"),
        "query_type": result.get("query_type", "unknown"),
        "sources": sources_list,
        "ai_enhanced": True
    }, indent=2)

@mcp.tool()
def list_available_sources() -> str:
    """List all sources."""
    lecture_files = sorted(set(t[0] for t in texts))
    dataset_info = [{"name": ds_name, "examples": len(ds_data["data"])} 
                    for ds_name, ds_data in DATASETS.items()]
    
    return json.dumps({
        "lectures": {
            "count": len(lecture_files),
            "files": lecture_files,
            "total_chunks": len(texts)
        },
        "datasets": dataset_info,
        "total_examples": len(texts) + sum(len(d['data']) for d in DATASETS.values())
    }, indent=2)

if __name__ == "__main__":
    mcp.run()