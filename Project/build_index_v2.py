import os
import re
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    """Embed a list of texts locally using sentence-transformers"""
    return embedder.encode(texts, convert_to_numpy=True).astype("float32")

def clean_text(text):
    """Clean up extracted text from PDFs with better artifact removal"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove standalone page numbers and headers/footers
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page \d+', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove common PDF artifacts (but preserve meaningful content)
    # Remove isolated special characters that are likely artifacts
    text = re.sub(r'\s+[•·▪▫]\s+', ' ', text)  # Bullet points
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Remove excessive punctuation (but keep sentence endings)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Clean up line breaks that shouldn't be there
    # If lowercase letter followed by uppercase, likely missing period
    text = re.sub(r'([a-z])\s+([A-Z][a-z])', r'\1. \2', text)
    
    # Remove mathematical notation artifacts that are common in PDFs
    # These are often OCR errors or formatting issues
    text = re.sub(r'𝑅[ଵଶଷ৪৫৬৭৮৯]', '', text)  # Remove subscript R with numbers
    text = re.sub(r'𝐸ே𝑌\|', 'E[Y|', text)  # Convert to readable format
    text = re.sub(r'[<≥≤]', '', text)  # Remove comparison operators that are artifacts
    
    # Remove isolated single characters (likely OCR errors)
    text = re.sub(r'\s+[A-Za-z]\s+', ' ', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_into_sentences(text):
    """Split text into sentences, handling abbreviations and decimal numbers"""
    # Pattern to split on sentence endings, but not on abbreviations or decimals
    # This regex handles: . ! ? but not if followed by lowercase or if it's a decimal
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    
    # First, protect abbreviations and decimal numbers
    # Replace common abbreviations temporarily
    abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'etc.', 'e.g.', 'i.e.', 'vs.', 'U.S.', 'U.K.']
    for abbr in abbreviations:
        text = text.replace(abbr, abbr.replace('.', '<DOT>'))
    
    # Protect decimal numbers
    text = re.sub(r'(\d+)\.(\d+)', r'\1<DOT>\2', text)
    
    # Now split on sentence endings
    sentences = re.split(sentence_endings, text)
    
    # Restore dots
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    
    # Clean up sentences
    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        # Filter out very short sentences (likely artifacts)
        if len(sent) > 10 and not sent.isdigit():
            cleaned_sentences.append(sent)
    
    return cleaned_sentences

def chunk_text_sentence_aware(text, max_words=250, min_words=50, overlap_sentences=2):
    """Split text into sentence-aware chunks for better retrieval"""
    # Split into sentences first
    sentences = split_into_sentences(text)
    
    if not sentences:
        # Fallback to word-based chunking if sentence splitting fails
        words = text.split()
        if len(words) < min_words:
            return [text] if len(text.strip()) > 30 else []
        chunks = []
        for i in range(0, len(words), max_words - 50):
            chunk = ' '.join(words[i:i + max_words])
            if len(chunk.strip()) > 30:
                chunks.append(chunk)
        return chunks
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max_words, save current chunk
        if current_word_count + sentence_words > max_words and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if current_word_count >= min_words:
                chunks.append(chunk_text)
            
            # Overlap: keep last N sentences for context
            if overlap_sentences > 0 and len(current_chunk) >= overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk = []
                current_word_count = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_word_count += sentence_words
        i += 1
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if current_word_count >= min_words or len(chunks) == 0:  # Always include last chunk
            chunks.append(chunk_text)
    
    # Filter out chunks that are too short
    return [chunk for chunk in chunks if len(chunk.strip()) > 30]

def chunk_text(text, chunk_size=300, overlap=50):
    """Legacy function - now uses sentence-aware chunking"""
    return chunk_text_sentence_aware(text, max_words=chunk_size, min_words=50, overlap_sentences=2)

def extract_metadata(text):
    """Extract key information like names, dates, topics"""
    metadata = []
    
    # Extract potential names (Title Case words, 2-4 words)
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\b'
    names = re.findall(name_pattern, text)
    for name in names[:5]:  # Top 5 potential names
        metadata.append(f"Person: {name}")
    
    # Extract dates
    date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
    dates = re.findall(date_pattern, text)
    for date in dates[:3]:
        metadata.append(f"Date: {date}")
    
    return " | ".join(metadata) if metadata else ""

def process_pdfs(folder="lectures", use_metadata=True):
    """Extract and chunk text from all PDFs with metadata enrichment"""
    texts = []
    if not os.path.exists(folder):
        print(f"Warning: {folder} folder not found. Creating it...")
        os.makedirs(folder)
        return texts
    
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {folder}/")
        return texts
    
    for file in pdf_files:
        print(f"Processing {file}...")
        reader = PdfReader(os.path.join(folder, file))
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text = clean_text(text)
                
                # Extract metadata for enrichment
                metadata = extract_metadata(text) if use_metadata else ""
                
                # Create smaller chunks
                chunks = chunk_text(text, chunk_size=300, overlap=50)
                
                for chunk_idx, chunk in enumerate(chunks):
                    # Clean the chunk further for better readability
                    clean_chunk = clean_text(chunk)
                    
                    # Optionally prepend metadata to chunk for better context
                    enriched_chunk = f"{metadata}\n{clean_chunk}" if metadata else clean_chunk
                    
                    # Store: (file, page, chunk_idx, enriched_for_embedding, clean_for_display)
                    texts.append((file, page_num, chunk_idx, enriched_chunk, clean_chunk))
    
    return texts

def build_index(texts):
    """Build FAISS index from extracted lecture texts"""
    if not texts:
        print("No texts to index!")
        return None, texts
    
    print(f"Building embeddings for {len(texts)} text chunks...")
    # Use enriched text for embedding
    embeddings = embed_texts([t[3] for t in texts])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, texts

if __name__ == "__main__":
    print("Building lecture note index with metadata enrichment...")
    print("-" * 50)
    
    texts = process_pdfs(use_metadata=True)
    
    if not texts:
        print("No content found. Please add lecture PDF files to the 'lectures' folder.")
        exit(1)
    
    index, texts = build_index(texts)
    
    if index:
        faiss.write_index(index, "lectures_v2.index")
        with open("lectures_v2.pkl", "wb") as f:
            pickle.dump(texts, f)
        print("-" * 50)
        print(f"✓ Lecture index built successfully!")
        print(f"  Total chunks: {len(texts)}")
        print(f"  Books processed: {len(set(t[0] for t in texts))}")
    else:
        print("Failed to build index.")