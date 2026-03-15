"""
Fase 1: Construir o índice RAG da Ética de Spinoza.

Carrega fontes (texto da Ética PT + axiomas Coq), chunka, gera embeddings,
e salva índice FAISS + mapping de chunks para retrieval posterior.

Uso:
    python build_index.py                    # build padrão
    python build_index.py --test "conatus"   # testa retrieval após build
"""

import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
INDEX_DIR = SCRIPT_DIR / "rag_index"

# Sources
ETHICS_PT = PROJECT_ROOT / "data" / "sources" / "etica_spinoza_pt.md"
ETHICS_COQ = PROJECT_ROOT / "sources" / "Ethics" / "Ethics.v"
PROMPT_FILE = PROJECT_ROOT / "data" / "prompts" / "prompt_geracao_spinoza.txt"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 300  # palavras
CHUNK_OVERLAP = 50  # palavras


def load_text_sources() -> list[dict]:
    """Carrega todas as fontes de texto e retorna como lista de documentos."""
    docs = []

    # 1. Texto da Ética em PT (principal)
    if ETHICS_PT.exists():
        text = ETHICS_PT.read_text(encoding="utf-8")
        docs.append(
            {"source": "etica_pt", "text": text, "label": "Ética de Spinoza (PT)"}
        )
        print(f"  📖 Ética PT: {len(text)} chars")

    # 2. Axiomas Coq (formal)
    if ETHICS_COQ.exists():
        text = ETHICS_COQ.read_text(encoding="utf-8", errors="replace")
        # Extrair apenas comentários (entre (** e *)) que são as descrições legíveis
        comments = re.findall(r"\(\*\*\s*(.*?)\s*\*\)", text, re.DOTALL)
        coq_text = "\n\n".join(comments)
        # Também pegar as linhas de definições/axiomas/hipóteses
        defs = re.findall(
            r"((?:Hypothesis|Axiom|Lemma|Theorem|Variable|Definition)\s+\w+\s*:.*?)\.",
            text,
            re.DOTALL,
        )
        coq_text += "\n\n" + "\n".join(defs)
        docs.append(
            {"source": "coq_proofs", "text": coq_text, "label": "Provas Formais (Coq)"}
        )
        print(f"  🔬 Coq: {len(coq_text)} chars")

    # 3. Prompt de geração (regras estilísticas)
    if PROMPT_FILE.exists():
        text = PROMPT_FILE.read_text(encoding="utf-8")
        docs.append(
            {"source": "prompt_rules", "text": text, "label": "Regras Estilísticas"}
        )
        print(f"  📝 Prompt: {len(text)} chars")

    return docs


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[dict]:
    """Divide texto em chunks com overlap."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if len(chunk_text.strip()) > 50:  # Ignorar chunks muito pequenos
            chunks.append(
                {
                    "text": chunk_text,
                    "source": source,
                    "word_start": start,
                    "word_end": min(end, len(words)),
                }
            )

        start += chunk_size - overlap

    return chunks


def build_index(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatIP:
    """Gera embeddings e constrói índice FAISS."""
    texts = [c["text"] for c in chunks]
    print(f"  🧮 Gerando embeddings para {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Inner product com embeddings normalizados = cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    return index


def save_index(index: faiss.IndexFlatIP, chunks: list[dict]):
    """Salva índice FAISS e metadata dos chunks."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # Salvar chunks sem embeddings (só texto e metadata)
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"  💾 Índice salvo em {INDEX_DIR}/")
    print(f"     • faiss.index: {index.ntotal} vetores de dim {index.d}")
    print(f"     • chunks.json: {len(chunks)} chunks")


def test_retrieval(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    top_k: int = 3,
):
    """Testa retrieval com uma query."""
    print(f'\n🔍 Query: "{query}"')
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(q_emb, dtype=np.float32), top_k)

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = chunks[idx]
        print(f"\n  [{rank + 1}] Score: {score:.4f} | Fonte: {chunk['source']}")
        # Mostrar primeiras 200 chars
        preview = chunk["text"][:200].replace("\n", " ")
        print(f"      {preview}...")


def main():
    parser = argparse.ArgumentParser(
        description="Constrói índice RAG da Ética de Spinoza"
    )
    parser.add_argument(
        "--test", type=str, help="Testar retrieval com uma query após o build"
    )
    args = parser.parse_args()

    print("━━━ Fase 1: Build RAG Index ━━━\n")

    # 1. Carregar fontes
    print("📚 Carregando fontes...")
    docs = load_text_sources()
    if not docs:
        print("❌ Nenhuma fonte encontrada!")
        return

    # 2. Chunkar
    print(f"\n✂️  Chunkando (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"], doc["source"], CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  {doc['label']}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"  Total: {len(all_chunks)} chunks")

    # 3. Embeddings + FAISS
    print(f"\n🤖 Carregando modelo de embeddings ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    index = build_index(all_chunks, model)

    # 4. Salvar
    print()
    save_index(index, all_chunks)

    # 5. Teste opcional
    if args.test:
        # Reload pra testar o fluxo completo
        test_retrieval(args.test, model, index, all_chunks)
    else:
        # Teste padrão
        test_retrieval("O que é o conatus?", model, index, all_chunks)
        test_retrieval(
            "Como lidar com a inveja segundo Spinoza?", model, index, all_chunks
        )

    print("\n✅ Índice RAG pronto!")


if __name__ == "__main__":
    main()
