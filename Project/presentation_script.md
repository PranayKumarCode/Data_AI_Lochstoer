# Presentation Script - 15 Minutes
## Lecture Assistant System

---

## Slide 1: Title & Overview (1 minute)

**Speaker Notes:**
- "Good [morning/afternoon]. Today I'll present our Lecture Assistant System, an AI-powered solution for searching and extracting information from course materials."
- "We built a Retrieval-Augmented Generation system that allows students to quickly find answers from 11 indexed lecture PDFs."
- "The system uses FAISS vector database, sentence transformers, and a local LLM to provide accurate, source-grounded answers."

**Key Points:**
- Problem: Quick access to lecture information
- Solution: RAG system
- Technology: FAISS + Sentence Transformers + Local LLM

---

## Slide 2: Problem Statement (1 minute)

**Speaker Notes:**
- "Let me start with the problem we're solving."
- "Students have 11 lecture PDFs with hundreds of pages. Finding specific information is time-consuming."
- "Traditional keyword search is limited - it doesn't understand semantic meaning."
- "We need a system that can understand questions and find relevant content quickly."

**Key Points:**
- Information overload (11 PDFs, 370+ pages)
- Need for semantic search
- Requirements: fast, accurate, local

---

## Slide 3: System Architecture (1.5 minutes)

**Speaker Notes:**
- "Here's our high-level architecture."
- "User queries go through the MCP server, which performs semantic search using FAISS."
- "The system searches through embedded lecture PDFs and returns relevant passages."
- "We have three main components: index builder, MCP server, and client interface."

**Key Points:**
- Three main components
- Query → Search → Answer flow
- MCP protocol for standardized access

---

## Slide 4: Data - Lecture Materials (1 minute)

**Speaker Notes:**
- "Our dataset consists of 11 lecture PDFs covering various AI and finance topics."
- "Total of approximately 370 pages of academic content."
- "Topics range from basic concepts like visualization to advanced topics like deep learning."
- "The content includes mathematical formulas, code examples, and conceptual explanations."

**Key Points:**
- 11 PDFs, ~370 pages
- Diverse topics (T1-T9)
- Academic content with formulas and code

---

## Slide 5: Data Preprocessing Pipeline (1.5 minutes)

**Speaker Notes:**
- "Let me explain how we process the PDFs."
- "First, we extract text using pypdf library."
- "Then we clean the text - removing whitespace and page numbers."
- "Most importantly, we chunk the text into 300-word segments with 50-word overlap."
- "This chunking strategy balances context and precision - crucial for semantic search."
- "We also extract metadata like names and dates for better context."

**Key Points:**
- PDF extraction → Cleaning → Chunking
- 300-word chunks, 50-word overlap
- Metadata extraction

---

## Slide 6: Embedding Methodology (1.5 minutes)

**Speaker Notes:**
- "For semantic search, we need to convert text into numerical vectors."
- "We use the all-MiniLM-L6-v2 model from Sentence Transformers."
- "This model converts each 300-word chunk into a 384-dimensional vector."
- "Why this model? It's fast - about 100 sentences per second - and accurate."
- "It's trained on over 1 billion sentence pairs, so it understands semantic relationships well."
- "The embeddings capture meaning, not just keywords."

**Key Points:**
- Model: all-MiniLM-L6-v2
- 384 dimensions
- Fast and accurate
- Semantic understanding

---

## Slide 7: Vector Database - FAISS (1.5 minutes)

**Speaker Notes:**
- "We store these embeddings in FAISS, Facebook's vector database."
- "FAISS uses L2 distance to find the most similar vectors."
- "When a user asks a question, we embed the query and search for the closest matches."
- "Lower distance means more similar content."
- "FAISS is extremely fast - it can search through thousands of vectors in milliseconds."
- "We use IndexFlatL2, which gives exact results, not approximations."

**Key Points:**
- FAISS for vector storage
- L2 distance metric
- Fast search (<100ms)
- Exact matching

---

## Slide 8: Retrieval-Augmented Generation (RAG) (1.5 minutes)

**Speaker Notes:**
- "Our system uses RAG - Retrieval-Augmented Generation."
- "Instead of just asking an LLM directly, we first retrieve relevant context from our lectures."
- "This ensures answers are grounded in the actual course materials."
- "The process: query expansion, semantic search, context retrieval, then answer extraction."
- "This prevents hallucination - the LLM can only answer based on retrieved content."

**Key Points:**
- RAG prevents hallucination
- Query → Search → Retrieve → Answer
- Grounded in course materials

---

## Slide 9: Answer Extraction Methodology (1.5 minutes)

**Speaker Notes:**
- "We have two methods for answer extraction."
- "Primary method uses a local LLM - Ollama with qwen2.5:32b model."
- "We use sophisticated prompt engineering to extract precise answers."
- "The prompt identifies question types - who, what, when, where - and extracts accordingly."
- "If the LLM isn't available, we fall back to regex-based extraction."
- "This ensures the system always works, even without AI capabilities."

**Key Points:**
- Two-stage extraction
- AI-powered (primary)
- Regex fallback
- Question type identification

---

## Slide 10: MCP Server (1 minute)

**Speaker Notes:**
- "Our system is built as an MCP server - Model Context Protocol."
- "This provides a standardized interface with four main tools."
- "Search lectures, ask simple questions, teach examples, and list available lectures."
- "MCP is language-agnostic and easy to integrate with other systems."

**Key Points:**
- MCP protocol
- Four tools
- Standardized interface

---

## Slide 11: Implementation Details (1 minute)

**Speaker Notes:**
- "Technical stack: Python with FAISS, Sentence Transformers, and MCP framework."
- "The index is about 825 KB, containing over 1,200 text chunks."
- "Build time is 2-3 minutes, but search is sub-second."
- "All code is modular and well-documented."

**Key Points:**
- Python stack
- 1,200+ chunks
- Fast search

---

## Slide 12: Query Processing Flow (1 minute)

**Speaker Notes:**
- "Let me walk through what happens when a user asks a question."
- "Query expansion adds synonyms for better retrieval."
- "Then embedding generation converts the query to a vector."
- "FAISS search finds the top 5 most similar chunks."
- "Answer extraction tries AI first, falls back to regex."
- "Total time: under 6 seconds, usually much faster."

**Key Points:**
- 6-step process
- <6 seconds total
- Multiple fallbacks

---

## Slide 13: Results & Performance (1 minute)

**Speaker Notes:**
- "Our system successfully searches across all 11 PDFs."
- "Relevance scores range from 0.7 to 1.3 - lower is better."
- "We tested various queries and consistently found relevant content."
- "The system covers all topics from basic concepts to advanced deep learning."

**Key Points:**
- High accuracy
- Good relevance scores
- Comprehensive coverage

---

## Slide 14: Demo Results (1 minute)

**Speaker Notes:**
- "Let me show you some actual results from our system."
- "For 'What is a decision tree?', we found content in T5, page 18, with 70% relevance."
- "For logistic regression, we found multiple relevant passages in T3."
- "The system successfully handles various question types."

**Key Points:**
- Real examples
- Multiple query types
- High relevance

---

## Slide 15: Limitations (30 seconds)

**Speaker Notes:**
- "Current limitations include requiring Ollama for AI extraction."
- "Some mathematical formulas may not extract perfectly."
- "Answer quality depends on chunk boundaries."
- "But the system is functional and provides good results."

**Key Points:**
- Ollama requirement
- Formula extraction
- Chunk boundaries

---

## Slide 16: Future Improvements (30 seconds)

**Speaker Notes:**
- "Future work includes better answer extraction, web interface, and GPU acceleration."
- "Long-term: multi-modal support for images and diagrams."
- "Advanced RAG techniques like re-ranking."

**Key Points:**
- Short-term improvements
- Long-term vision

---

## Slide 17: Comparison (30 seconds)

**Speaker Notes:**
- "Compared to alternatives, our system offers source grounding, privacy, and no costs."
- "Unlike ChatGPT, answers are always grounded in lecture materials."
- "Unlike cloud services, everything runs locally."

**Key Points:**
- Advantages over alternatives
- Privacy and cost benefits

---

## Slide 18: Technical Achievements (30 seconds)

**Speaker Notes:**
- "We built a complete RAG pipeline from scratch."
- "Production-ready system with error handling."
- "Scalable to 100+ PDFs."
- "Optimized for educational content."

**Key Points:**
- Complete system
- Production-ready
- Scalable

---

## Slide 19: Use Cases (30 seconds)

**Speaker Notes:**
- "Practical applications include student study assistance, TA tools, and research support."
- "Saves time and improves learning efficiency."

**Key Points:**
- Multiple use cases
- Educational impact

---

## Slide 20: Conclusion (30 seconds)

**Speaker Notes:**
- "In summary, we built a working RAG system for lecture materials."
- "It provides fast, accurate, source-grounded answers."
- "The system is scalable and extensible."
- "Thank you for your attention."

**Key Points:**
- Problem solved
- Working system
- Future potential

---

## Slide 21: Q&A (Remaining time)

**Speaker Notes:**
- "I'm happy to answer any questions."
- "Code and documentation are available."
- "Thank you!"

---

## Timing Summary

1. Title (1 min)
2. Problem (1 min)
3. Architecture (1.5 min)
4. Data (1 min)
5. Preprocessing (1.5 min)
6. Embeddings (1.5 min)
7. FAISS (1.5 min)
8. RAG (1.5 min)
9. Answer Extraction (1.5 min)
10. MCP Server (1 min)
11. Implementation (1 min)
12. Query Flow (1 min)
13. Results (1 min)
14. Demo (1 min)
15. Limitations (0.5 min)
16. Future (0.5 min)
17. Comparison (0.5 min)
18. Achievements (0.5 min)
19. Use Cases (0.5 min)
20. Conclusion (0.5 min)
21. Q&A (remaining)

**Total: ~15 minutes**

