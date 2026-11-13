# Lecture Assistant System
## AI-Powered RAG for Educational Content

**15-Minute Presentation**

---

## Slide 1: Title & Overview

### Lecture Assistant System
**Retrieval-Augmented Generation (RAG) for Course Materials**

- **Problem**: Students need quick access to information from lecture PDFs
- **Solution**: AI-powered semantic search with answer extraction
- **Technology**: FAISS vector database + Sentence Transformers + Local LLM
- **Result**: Instant answers from 11 indexed lecture PDFs

---

## Slide 2: Problem Statement

### Challenges in Educational Content Access

1. **Information Overload**
   - 11 lecture PDFs with hundreds of pages
   - Difficult to find specific information quickly
   - Time-consuming manual search

2. **Knowledge Retrieval**
   - Students ask questions during study sessions
   - Need accurate, source-grounded answers
   - Traditional keyword search is limited

3. **Solution Requirements**
   - Fast semantic search across all materials
   - Accurate answer extraction
   - Source citation for verification
   - Local deployment (privacy & cost)

---

## Slide 3: System Architecture Overview

### High-Level Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   MCP Server (mcp_server)   │
│  - Query Processing          │
│  - Answer Extraction         │
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  FAISS Vector Index          │
│  - Semantic Search           │
│  - Similarity Matching        │
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Lecture PDFs (11 files)     │
│  - Text Extraction           │
│  - Chunking & Embedding      │
└─────────────────────────────┘
```

**Components:**
- **Index Builder**: Processes PDFs → Embeddings → FAISS Index
- **MCP Server**: Handles queries and answer extraction
- **Client**: Test interface for queries

---

## Slide 4: Data - Lecture Materials

### Dataset Overview

**Source**: Course lecture PDFs (MFE-431 AI in Finance)

| Lecture | Topic | Pages |
|---------|-------|-------|
| T1 | Introductory, Visualization | ~20 |
| T2 | Panel Regressions | ~25 |
| T3 | Logistic Regressions, FinTech | ~30 |
| T4 | Model Selection and Shrinkage | ~25 |
| T5 | Decision Trees, Boosting, Bagging | ~45 |
| T6 | Textual Analysis, Sentiment | ~50 |
| T6b | Textual Analysis, Mergers | ~30 |
| T7 | Unstructured Data | ~40 |
| T7b | Large Language Models | ~35 |
| T8 | Support Vector Machines | ~30 |
| T9 | Deep Learning | ~40 |

**Total**: ~370 pages of academic content

**Content Type**: 
- Mathematical formulas
- Code examples
- Conceptual explanations
- Financial applications

---

## Slide 5: Data Preprocessing Pipeline

### Step-by-Step Processing

#### 1. **PDF Text Extraction**
```python
# Using pypdf library
reader = PdfReader(pdf_file)
for page in reader.pages:
    text = page.extract_text()
```

#### 2. **Text Cleaning**
- Remove excessive whitespace
- Remove page numbers
- Normalize formatting
- Preserve mathematical notation

#### 3. **Text Chunking Strategy**
- **Chunk Size**: 300 words
- **Overlap**: 50 words
- **Rationale**: 
  - Balance context vs. precision
  - Overlap ensures no information loss at boundaries
  - Optimal for semantic search

#### 4. **Metadata Extraction**
- Extract names (professors, authors)
- Extract dates (deadlines, schedules)
- File name and page number tracking

---

## Slide 6: Embedding Methodology

### Sentence Transformers for Semantic Search

**Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Type**: Sentence-level embeddings
- **Training**: Trained on 1B+ sentence pairs
- **Speed**: ~100 sentences/second

**Why This Model?**
1. **Fast**: Optimized for speed
2. **Accurate**: Good semantic understanding
3. **Lightweight**: Small model size (~80MB)
4. **General Purpose**: Works well on academic text

**Embedding Process**:
```python
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, convert_to_numpy=True)
# Output: (n_chunks, 384) numpy array
```

**Result**: Each text chunk → 384-dimensional vector

---

## Slide 7: Vector Database - FAISS

### FAISS Index Structure

**Index Type**: `IndexFlatL2` (Exact L2 distance)

**Why FAISS?**
- **Fast**: Optimized C++ implementation
- **Scalable**: Handles millions of vectors
- **Efficient**: GPU support available
- **Industry Standard**: Used by Facebook/Meta

**Index Building**:
```python
dim = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dim)
index.add(embeddings)  # Add all chunk embeddings
```

**Search Process**:
```python
query_embedding = embedder.encode([query])
distances, indices = index.search(query_embedding, k=5)
# Returns top k most similar chunks
```

**Distance Metric**: L2 (Euclidean) distance
- Lower distance = More similar
- Range: 0 (identical) to ∞ (different)

---

## Slide 8: Retrieval-Augmented Generation (RAG)

### RAG Architecture

**Traditional LLM**:
```
Query → LLM → Answer (may hallucinate)
```

**RAG Approach**:
```
Query → Embed → Search → Retrieve Context → LLM → Grounded Answer
```

**Our Implementation**:

1. **Query Expansion**
   - Add synonyms for better retrieval
   - Example: "TA" → "teaching assistant contact office hours"

2. **Semantic Search**
   - Find top-k relevant passages
   - Rank by similarity score

3. **Answer Extraction** (Two Methods)
   - **Primary**: Local LLM (Ollama qwen2.5:32b)
   - **Fallback**: Regex-based extraction

4. **Source Citation**
   - Return file name and page number
   - Enable fact verification

---

## Slide 9: Answer Extraction Methodology

### Two-Stage Answer Extraction

#### Stage 1: AI-Powered Extraction (Primary)

**LLM**: Ollama qwen2.5:32b (Local)
- **Model Size**: 32 billion parameters
- **Deployment**: Local (no API costs)
- **Privacy**: Data never leaves local machine

**Prompt Engineering**:
```
You are an expert information extraction AI.
Given a question and search results, extract ONLY the specific answer.

CRITICAL THINKING PROCESS:
1. Read question carefully
2. Examine ALL search results
3. Find explicit statements
4. Return ONLY precise answer

QUESTION TYPE IDENTIFICATION:
- "Who is..." → Extract person names
- "When is..." → Extract dates/times
- "What is..." → Extract concepts
- "How many..." → Extract numbers
```

**Output**: Direct answer + confidence score

#### Stage 2: Regex Fallback

**When AI unavailable**:
- Pattern matching for names, dates, numbers
- Simple extraction rules
- Returns "Not found" if no match

---

## Slide 10: MCP (Model Context Protocol) Server

### Server Architecture

**Framework**: FastMCP (Python)

**Available Tools**:

1. **`search_lectures(query, k, use_ai)`**
   - Semantic search with answer extraction
   - Returns: passages + extracted answer

2. **`ask_simple_question(question)`**
   - Direct Q&A interface
   - Returns: concise answer + sources

3. **`teach_example(question, answer, explanation)`**
   - Add training examples
   - Improves answer quality via few-shot learning

4. **`list_lectures()`**
   - List all indexed PDF files

**Communication**: stdio-based MCP protocol
- Language-agnostic
- Standardized interface
- Easy integration

---

## Slide 11: Implementation Details

### Technical Stack

**Languages & Libraries**:
- Python 3.8+
- FAISS-CPU 1.13.0 (vector search)
- Sentence-Transformers 5.1.2 (embeddings)
- PyPDF 6.4.0 (PDF processing)
- MCP 1.23.1 (server framework)

**File Structure**:
```
Project/
├── build_index_v2.py      # Index builder
├── mcp_server_v2.py       # MCP server
├── client_test_v2.py       # Test client
├── teach_ai.py            # Training script
├── ai_config.json         # Configuration
├── lectures/              # PDF files
├── lectures_v2.index      # FAISS index (825 KB)
└── lectures_v2.pkl        # Text chunks (457 KB)
```

**Index Statistics**:
- Total chunks: ~1,200+
- Index size: 825 KB
- Build time: ~2-3 minutes
- Search latency: <100ms

---

## Slide 12: Query Processing Flow

### Detailed Query Flow

```
1. User Query: "What is a decision tree?"
   │
   ▼
2. Query Expansion
   - Original: "decision tree"
   - Expanded: ["decision tree", "tree model", "classification tree"]
   │
   ▼
3. Embedding Generation
   - Query → 384-dim vector
   │
   ▼
4. FAISS Search
   - Compare with all chunks
   - Return top 5 matches
   │
   ▼
5. Answer Extraction
   ├─ Try AI extraction (if Ollama available)
   └─ Fallback to regex
   │
   ▼
6. Response Formatting
   - Answer + confidence
   - Source citations
   - Relevance scores
```

**Performance**:
- Search: ~50-100ms
- AI extraction: ~2-5 seconds (if available)
- Total: <6 seconds

---

## Slide 13: Results & Performance

### System Capabilities

**Search Accuracy**:
- Successfully finds relevant content across all 11 PDFs
- Relevance scores range from 0.7 (high) to 1.3 (low)
- Top results consistently match query intent

**Example Queries**:

| Query | Found In | Relevance |
|-------|----------|-----------|
| "What is a decision tree?" | T5, Page 18 | 70% |
| "Explain boosting" | T5, Page 39 | 89% |
| "Logistic regression" | T3, Pages 3-5 | 79-85% |
| "Sentiment analysis" | T6, Page 45 | 95% |

**Coverage**:
- ✅ All 11 lecture PDFs indexed
- ✅ 370+ pages searchable
- ✅ Mathematical formulas preserved
- ✅ Code examples included

---

## Slide 14: Demo Results

### Live System Demonstration

**Available Lectures**: 11 PDFs indexed

**Test Queries**:

1. **"What is a decision tree?"**
   - Found: T5 - Decision Trees (Page 18)
   - Content: "A fitted tree with three boxes... easy to interpret"
   - Relevance: 70%

2. **"How does logistic regression work?"**
   - Found: T3 - Logistic Regressions (Pages 3, 5)
   - Content: "Binary dependent variable... probability that Y=1"
   - Relevance: 79-85%

3. **"What is sentiment analysis?"**
   - Found: T6 - Textual Analysis (Page 45)
   - Content: "Create a Sentiment Indicator... Baker and Wurgler"
   - Relevance: 95%

**System Status**: ✅ Operational**

---

## Slide 15: Limitations & Challenges

### Current Limitations

1. **AI Answer Extraction**
   - Requires Ollama installation
   - Needs qwen2.5:32b model (~20GB)
   - Falls back to regex if unavailable

2. **Answer Quality**
   - Without AI: Returns "Not found"
   - With AI: More accurate extraction
   - Regex limited to simple patterns

3. **Mathematical Formulas**
   - PDF extraction may miss complex formulas
   - LaTeX notation not always preserved
   - Some symbols may be lost

4. **Context Window**
   - Chunks are 300 words
   - May split long explanations
   - Overlap helps but not perfect

5. **Language**
   - Optimized for English
   - May struggle with mixed languages

---

## Slide 16: Future Improvements

### Enhancement Roadmap

**Short-term**:
1. **Better Answer Extraction**
   - Improve regex patterns
   - Add more question types
   - Better error handling

2. **User Interface**
   - Web interface for queries
   - Interactive chat interface
   - Visual search results

3. **Performance**
   - GPU acceleration for FAISS
   - Caching frequent queries
   - Parallel processing

**Long-term**:
1. **Multi-modal Support**
   - Image extraction from PDFs
   - Diagram understanding
   - Table extraction

2. **Advanced RAG**
   - Re-ranking with cross-encoder
   - Query rewriting
   - Context compression

3. **Learning System**
   - User feedback integration
   - Continuous improvement
   - Personalized results

---

## Slide 17: Comparison with Alternatives

### Why This Approach?

| Feature | Our System | Traditional Search | ChatGPT |
|---------|------------|-------------------|---------|
| **Source Grounding** | ✅ Always | ❌ No | ⚠️ Sometimes |
| **Privacy** | ✅ Local | ✅ Local | ❌ Cloud |
| **Cost** | ✅ Free | ✅ Free | ❌ Paid |
| **Accuracy** | ✅ High | ⚠️ Medium | ⚠️ Variable |
| **Speed** | ✅ Fast | ✅ Fast | ⚠️ Slower |
| **Customization** | ✅ Full | ❌ Limited | ❌ Limited |

**Key Advantages**:
- Grounded in lecture materials (no hallucination)
- Privacy-preserving (local deployment)
- No API costs
- Customizable for specific course content

---

## Slide 18: Technical Achievements

### What We Built

1. **Complete RAG Pipeline**
   - PDF → Text → Embeddings → Index → Search → Answer

2. **Production-Ready System**
   - MCP server for standardized access
   - Error handling and fallbacks
   - Configuration management

3. **Scalable Architecture**
   - Can handle 100+ PDFs
   - Efficient vector search
   - Modular design

4. **Educational Focus**
   - Optimized for academic content
   - Preserves mathematical notation
   - Source citation for verification

**Code Quality**:
- Clean, documented code
- Modular design
- Easy to extend

---

## Slide 19: Use Cases

### Practical Applications

1. **Student Study Assistant**
   - Quick answers during study sessions
   - Review before exams
   - Concept clarification

2. **Teaching Assistant Tool**
   - Answer common student questions
   - Reference material lookup
   - Content verification

3. **Research Support**
   - Literature review assistance
   - Concept exploration
   - Cross-reference finding

4. **Course Material Navigation**
   - Find specific topics quickly
   - Discover related concepts
   - Explore lecture content

**Impact**: Saves time, improves learning efficiency

---

## Slide 20: Conclusion

### Key Takeaways

1. **Problem Solved**
   - ✅ Fast semantic search across lecture PDFs
   - ✅ Accurate answer extraction
   - ✅ Source-grounded responses

2. **Technology Stack**
   - FAISS for vector search
   - Sentence Transformers for embeddings
   - MCP for standardized interface
   - Local LLM for answer extraction

3. **Results**
   - 11 PDFs indexed (~370 pages)
   - Sub-second search latency
   - High relevance matching

4. **Future Potential**
   - Scalable to larger document sets
   - Extensible architecture
   - Multiple use cases

**Thank You!**

---

## Slide 21: Q&A

### Questions?

**Contact & Resources**:
- Code: Available in Project directory
- Documentation: README.md
- Demo: `python3 demo.py`

**Key Files**:
- `build_index_v2.py` - Index builder
- `mcp_server_v2.py` - Server implementation
- `client_test_v2.py` - Test client

**Thank you for your attention!**

