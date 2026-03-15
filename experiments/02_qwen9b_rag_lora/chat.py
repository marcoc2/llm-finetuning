import argparse
import sys
import json
import numpy as np
from pathlib import Path

# Unsloth e libs de RAG
from unsloth import FastLanguageModel
import faiss
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).parent
ADAPTER_DIR = SCRIPT_DIR / "adapter"
INDEX_DIR = SCRIPT_DIR / "rag_index"


# 1. Carregar Índice RAG
def load_rag():
    print("⏳ Carregando RAG (SentenceTransformer + FAISS)...")
    if not (INDEX_DIR / "faiss.index").exists():
        print("❌ Índice RAG não encontrado! Execute build_index.py primeiro.")
        sys.exit(1)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return model, index, chunks


def retrieve_context(query: str, model, index, chunks, top_k=3):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(q_emb, dtype=np.float32), top_k)

    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        # Filtra threshold mínimo para evitar contexto inútil
        if score > 0.3:
            retrieved.append(chunks[idx]["text"])

    return "\n\n".join(retrieved)


# 2. Carregar Modelo Qwen com LoRA
def load_llm():
    print("⏳ Carregando Qwen 3.5-9B + LoRA Adapter...")
    if not ADAPTER_DIR.exists():
        print(
            "⚠️ Adaptador não encontrado. Carregando modelo base apenas (sem fine-tuning)."
        )
        model_name = "Qwen/Qwen3.5-9B"
    else:
        model_name = str(ADAPTER_DIR)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# Formato prompt Qwen + contexto
chatml_prompt = """<|im_start|>system
Você é um filósofo rigoroso, baseado exclusivamente na obra "Ética" de Baruch Spinoza.
Responda adotando um tom professoral, racionalista e estruturado.
Utilize o CONTEXTO DA ÉTICA fornecido para construir seu raciocínio. Ignore informações externas que o contradigam.

=== CONTEXTO DA ÉTICA ===
{context_texts}
<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
"""

# Ou caso a Lora tenha sido treinada no formato Alpaca, podemos manter o alpaca_prompt com o Contexto
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Aja como um spinozista. Analise a situação de acordo com Spinoza, baseando-se RIGOROSAMENTE nos seguintes trechos da Ética:

=== TRECHOS DA ÉTICA ===
{context_texts}
========================

### Input:
{user_query}

### Output:
"""


def generate_response(model, tokenizer, context, query):
    prompt = alpaca_prompt.format(context_texts=context, user_query=query)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, max_new_tokens=512, use_cache=True, temperature=0.7
    )
    # Extrair apenas a saída do assistente
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Isolar a resposta após o Output:
    if "### Output:\n" in decoded:
        response = decoded.split("### Output:\n")[1]
    else:
        response = decoded

    return response


def main():
    parser = argparse.ArgumentParser(description="Spinoza RAG + LoRA Chat")
    parser.add_argument(
        "--query", type=str, required=True, help="O dilema humano para Spinoza analisar"
    )
    args = parser.parse_args()

    # RAG
    embed_model, faiss_index, chunks = load_rag()
    context = retrieve_context(args.query, embed_model, faiss_index, chunks)

    # LLM
    llm_model, tokenizer = load_llm()

    print("\n" + "=" * 50)
    print(f"🧐 SUA PERGUNTA:\n{args.query}")
    print("\n📚 CONTEXTO RECUPERADO DA ÉTICA:")
    print(context[:300] + "...\n" if context else "Nenhum contexto forte encontrado.")

    print("\n🧠 RESPOSTA DE SPINOZA:")
    response = generate_response(llm_model, tokenizer, context, args.query)
    print(response)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
