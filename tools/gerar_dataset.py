"""
Gera entradas para o dataset_spinoza.jsonl usando LLM local via llama.cpp.

Compõe o prompt de forma modular:
  - System: prompt base (prompt_geracao_spinoza.txt) + categoria temática
  - User: amostra anti-repetição + instrução de geração

Uso:
    python gerar_dataset.py                          # 1 batch, categoria aleatória
    python gerar_dataset.py --batches 10             # 10 batches, rotação de categorias
    python gerar_dataset.py --categoria familia      # forçar categoria
    python gerar_dataset.py --verbose                # mostra prompts enviados
"""

import argparse
import difflib
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROMPT_FILE = PROJECT_ROOT / "data" / "prompts" / "prompt_geracao_spinoza.txt"
DATASET_FILE = PROJECT_ROOT / "data" / "datasets" / "spinoza_etica.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"

DEFAULT_URL = "http://localhost:8080"
REQUIRED_KEYS = {"instruction", "input", "output"}
SIMILARITY_THRESHOLD = 0.80

# ─── Categorias temáticas ────────────────────────────────────────────────────

CATEGORIAS = {
    "trabalho": {
        "nome": "Trabalho e Carreira",
        "desc": "demissão, burnout, promoção injusta, assédio moral, fracasso profissional, "
        "competição no trabalho, mudança de carreira, desemprego, sobrecarga",
    },
    "relacionamentos": {
        "nome": "Relacionamentos Amorosos",
        "desc": "ciúme, traição, término, dependência emocional, relacionamento à distância, "
        "amor não correspondido, medo de compromisso, manipulação afetiva",
    },
    "familia": {
        "nome": "Família e Parentalidade",
        "desc": "conflito com pais, criação de filhos, irmãos rivais, cuidar de idosos, "
        "herança, pressão familiar, expectativas dos pais, abandono familiar",
    },
    "saude": {
        "nome": "Saúde, Corpo e Vícios",
        "desc": "doença crônica, vício em substâncias, imagem corporal, transtornos alimentares, "
        "diagnóstico grave, ansiedade, insônia, hipocondria, luto por saúde perdida",
    },
    "dinheiro": {
        "nome": "Dinheiro e Consumismo",
        "desc": "dívidas, ganância, consumismo, apostas, comparação material, pobreza, "
        "medo de perder dinheiro, ostentação, exploração financeira",
    },
    "sociedade": {
        "nome": "Sociedade e Política",
        "desc": "injustiça social, polarização política, discriminação, corrupção, "
        "revolta contra o sistema, fanatismo ideológico, censura, desigualdade",
    },
    "tecnologia": {
        "nome": "Tecnologia e Mundo Digital",
        "desc": "vício em redes sociais, cyberbullying, privacidade digital, IA substituindo empregos, "
        "isolamento digital, comparação online, cancelamento, dependência de validação virtual",
    },
    "educacao": {
        "nome": "Educação e Conhecimento",
        "desc": "fracasso acadêmico, síndrome do impostor, pressão por notas, "
        "comparação com colegas, escolha de carreira, desistência dos estudos, autodidatismo",
    },
    "existencial": {
        "nome": "Questões Existenciais",
        "desc": "medo da morte, sentido da vida, solidão profunda, crise de meia-idade, "
        "vazio existencial, culpa existencial, envelhecimento, insignificância cósmica",
    },
    "moralidade": {
        "nome": "Moralidade e Ética Pessoal",
        "desc": "mentira, culpa, perdão, vingança, dilemas éticos, traição de valores, "
        "corrupção pessoal, hipocrisia, justiça vs. misericórdia",
    },
    "criatividade": {
        "nome": "Criatividade e Arte",
        "desc": "bloqueio criativo, medo de se expressar, crítica destrutiva à obra, "
        "comparação com artistas melhores, frustração por não viver de arte, perfeccionismo artístico, "
        "plágio, perda de inspiração, arte como fuga vs. arte como ofício",
    },
    "amizade": {
        "nome": "Amizade e Convívio Social",
        "desc": "amizade que esfriou, sentir-se excluído do grupo, falsidade de amigos, "
        "dificuldade de fazer amigos novos, amigo que só procura quando precisa, "
        "rivalidade entre amigos, fofoca, solidão social",
    },
    "sexualidade": {
        "nome": "Sexualidade e Intimidade",
        "desc": "insegurança sexual, disfunção, comparação corporal na intimidade, "
        "medo de rejeição física, conflito entre desejo e moral, "
        "descoberta da orientação sexual, pressão por performance, asexualidade incompreendida",
    },
    "envelhecimento": {
        "nome": "Envelhecimento e Finitude",
        "desc": "aposentadoria sem propósito, perda de amigos pela idade, corpo que falha, "
        "dependência dos filhos, medo de ser esquecido, nostalgia paralisante, "
        "sabedoria ignorada pelos jovens, solidão na velhice",
    },
    "espiritualidade": {
        "nome": "Espiritualidade e Religião",
        "desc": "perda da fé, conflito entre ciência e religião, fanatismo religioso, "
        "culpa religiosa, crise espiritual, medo do inferno, "
        "pressão da comunidade religiosa, ateísmo vs. busca de sentido",
    },
    "esporte": {
        "nome": "Esporte e Competição",
        "desc": "derrota humilhante, lesão que encerra carreira, obsessão por vitória, "
        "comparação com atletas melhores, pressão de treinador, doping, "
        "frustração por não evoluir, torcida agressiva, excesso de treino",
    },
    "justica": {
        "nome": "Justiça e Sistema Legal",
        "desc": "ser vítima de injustiça judicial, impunidade, processo longo e desgastante, "
        "testemunhar crime sem poder agir, condenação injusta, "
        "dilema entre legalidade e moralidade, vingança vs. justiça formal",
    },
    "migracao": {
        "nome": "Migração e Desenraizamento",
        "desc": "saudade da terra natal, xenofobia, barreira linguística, "
        "perda de identidade cultural, família dividida pela distância, "
        "sentir-se estrangeiro em toda parte, recomeçar do zero, refúgio",
    },
    "adolescencia": {
        "nome": "Adolescência e Juventude",
        "desc": "pressão dos pares, bullying, crise de identidade juvenil, "
        "conflito com autoridade, primeiro amor e rejeição, "
        "pressão por vestibular, automutilação, influência de ídolos tóxicos",
    },
    "luto": {
        "nome": "Luto e Perda",
        "desc": "morte de pai ou mãe, perda de filho, luto por animal de estimação, "
        "luto por amizade morta, perda de emprego como luto, "
        "luto antecipatório, culpa do sobrevivente, não conseguir chorar",
    },
    "identidade": {
        "nome": "Identidade e Autoimagem",
        "desc": "não saber quem é, viver conforme expectativas alheias, "
        "mudança de personalidade após trauma, conflito entre quem é e quem quer ser, "
        "identidade de gênero, crise de autenticidade, máscara social permanente",
    },
    "disciplina": {
        "nome": "Autodisciplina e Hábitos",
        "desc": "procrastinação crônica, vício em dopamina rápida (jogos, scrolling), "
        "incapacidade de manter rotinas, desistir de dietas, "
        "culpa por improdutividade, self-sabotage, perfeccionismo paralisante",
    },
    "meioambiente": {
        "nome": "Meio Ambiente e Natureza",
        "desc": "eco-ansiedade, impotência diante da crise climática, "
        "culpa por consumo não sustentável, conflito entre conforto e consciência ambiental, "
        "desmatamento da terra natal, luto ecológico, negacionismo climático",
    },
    "comunicacao": {
        "nome": "Comunicação e Conflito",
        "desc": "incapacidade de expressar sentimentos, mal-entendidos crônicos, "
        "briga que saiu do controle, comunicação passivo-agressiva, "
        "medo de confronto, mentira por omissão, silêncio punitivo, ghosting",
    },
    "lazer": {
        "nome": "Lazer, Prazer e Ócio",
        "desc": "culpa por descansar, vício em entretenimento, incapacidade de relaxar, "
        "hedonismo vazio, férias que não satisfazem, "
        "pressão para ser produtivo o tempo todo, prazer que vira compulsão, tédio no tempo livre",
    },
}


# ─── Funções ──────────────────────────────────────────────────────────────────


def load_base_prompt() -> str:
    """Carrega o prompt base fixo (axiomas Coq + formato JSONL + regras)."""
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def build_system_prompt(base: str, categoria_key: str) -> str:
    """Compõe system prompt = base + foco na categoria."""
    cat = CATEGORIAS[categoria_key]
    foco = (
        f"\n\nTEMA OBRIGATÓRIO DESTE BATCH: {cat['nome']}\n"
        f"Exemplos de situações nesta categoria: {cat['desc']}.\n"
        f"Gere as 10 situações focando EXCLUSIVAMENTE em dilemas de {cat['nome']}.\n"
        f"Seja criativo e cubra sub-temas variados dentro desta categoria."
    )
    return base + foco


def build_user_message(existing_inputs: list[str], categoria_key: str) -> str:
    """Compõe user message = anti-repetição + instrução."""
    cat = CATEGORIAS[categoria_key]
    parts = []

    # Amostra anti-repetição: até 5 inputs existentes aleatórios
    if existing_inputs:
        sample_size = min(5, len(existing_inputs))
        sample = random.sample(existing_inputs, sample_size)
        parts.append("Situações JÁ GERADAS (NÃO repita nenhuma parecida com estas):")
        for inp in sample:
            parts.append(f'- "{inp}"')
        parts.append("")

    parts.append(
        f"Gere agora 10 situações e saídas INÉDITAS sobre {cat['nome']}. "
        f"Cada situação deve ser diferente das anteriores. Apenas JSONL, sem markdown:"
    )
    return "\n".join(parts)


def call_llm(
    base_url: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
    verbose: bool,
) -> str:
    """Faz chamada à API do llama.cpp (/v1/chat/completions)."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    if verbose:
        print("\n  ── SYSTEM PROMPT ──")
        print(f"  {system_prompt[:300]}...")
        print("\n  ── USER MESSAGE ──")
        print(f"  {user_message}")
        print("  ──────────────────\n")

    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Não consegui conectar em {url}")
        print("   Inicie o llama-server primeiro:")
        print(r"   F:\workspace\llama_cpp\launch_server.bat")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Erro na chamada da API: {exc}")
        sys.exit(1)


def _try_fix_json(text: str) -> dict | None:
    """Tenta consertar JSON com aspas internas não escapadas via extração de campos."""
    try:
        idx_inst = text.find('"instruction"')
        idx_inp = text.find('"input"')
        idx_out = text.find('"output"')
        if idx_inst == -1 or idx_inp == -1 or idx_out == -1:
            return None

        # Extrair valores entre os delimitadores de campo
        def extract_value(text: str, key_end: int, next_key: int | None) -> str:
            # Pula ": " após a chave
            start = text.find('"', key_end + len('"')) + 1
            if next_key is not None:
                # Achar o último '",' antes da próxima chave
                segment = text[start:next_key]
                # Remover trailing '", ' ou '",'
                segment = segment.rstrip()
                if segment.endswith('",'):
                    segment = segment[:-2]
                elif segment.endswith('"'):
                    segment = segment[:-1]
            else:
                segment = text[start:]
                segment = segment.rstrip()
                if segment.endswith('"}'):
                    segment = segment[:-2]
                elif segment.endswith('"'):
                    segment = segment[:-1]
            return segment

        instruction = extract_value(text, idx_inst, idx_inp)
        input_val = extract_value(text, idx_inp, idx_out)
        output_val = extract_value(text, idx_out, None)

        return {"instruction": instruction, "input": input_val, "output": output_val}
    except Exception:
        return None


def parse_jsonl_lines(raw_text: str) -> list[dict]:
    """Extrai objetos JSON da resposta da LLM, mesmo multi-linha."""
    valid = []
    # Limpar markdown code blocks
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"```", "", cleaned)

    # Reconstruir objetos JSON rastreando profundidade de chaves
    depth = 0
    current = []
    in_string = False
    escape_next = False

    for ch in cleaned:
        if escape_next:
            current.append(ch)
            escape_next = False
            continue

        if ch == "\\" and in_string:
            current.append(ch)
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            current.append(ch)
            continue

        if in_string:
            # Substituir newlines literais dentro de strings por espaço
            if ch in ("\n", "\r"):
                current.append(" ")
                continue
            current.append(ch)
            continue

        if ch == "{":
            if depth == 0:
                current = []
            depth += 1
            current.append(ch)
        elif ch == "}":
            current.append(ch)
            depth -= 1
            if depth == 0:
                blob = "".join(current).strip()
                # Tentar parsear diretamente
                try:
                    obj = json.loads(blob)
                    if REQUIRED_KEYS.issubset(obj.keys()):
                        valid.append(obj)
                    else:
                        print(f"  ⚠ Campos faltando: {list(obj.keys())}")
                except json.JSONDecodeError:
                    # Tentar consertar aspas internas
                    fixed = _try_fix_json(blob)
                    if fixed and REQUIRED_KEYS.issubset(fixed.keys()):
                        valid.append(fixed)
                    else:
                        print(f"  ⚠ JSON irrecuperável: {blob[:80]}...")
                current = []
        elif depth > 0:
            current.append(ch)

    return valid


def is_similar(new_input: str, existing: list[str], threshold: float) -> bool:
    """Checa se new_input é similar demais a algum existente (fuzzy dedup)."""
    new_lower = new_input.strip().lower()
    for ex in existing:
        ratio = difflib.SequenceMatcher(None, new_lower, ex.strip().lower()).ratio()
        if ratio >= threshold:
            return True
    return False


def load_existing_inputs() -> list[str]:
    """Carrega todos os inputs existentes no dataset."""
    inputs = []
    if DATASET_FILE.exists():
        for line in DATASET_FILE.read_text(encoding="utf-8").strip().splitlines():
            try:
                obj = json.loads(line)
                inp = obj.get("input", "").strip()
                if inp:
                    inputs.append(inp)
            except json.JSONDecodeError:
                continue
    return inputs


def log_raw_response(
    batch_num: int, categoria: str, raw: str, parsed_count: int
) -> None:
    """Salva a resposta bruta da LLM em logs/ para análise posterior."""
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"batch_{batch_num:03d}_{categoria}_{timestamp}.txt"
    log_path = LOG_DIR / filename
    header = (
        f"Batch: {batch_num}\n"
        f"Categoria: {categoria}\n"
        f"Timestamp: {timestamp}\n"
        f"JSONL válidos parseados: {parsed_count}\n"
        f"{'═' * 60}\n"
    )
    log_path.write_text(header + raw, encoding="utf-8")


def append_to_dataset(entries: list[dict]) -> int:
    with open(DATASET_FILE, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(entries)


def count_lines() -> int:
    if not DATASET_FILE.exists():
        return 0
    return len(
        [
            line
            for line in DATASET_FILE.read_text(encoding="utf-8").strip().splitlines()
            if line.strip()
        ]
    )


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Gera dataset Spinoza via llama.cpp local"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL base do llama-server (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=1,
        help="Quantos batches de 10 gerar (default: 1)",
    )
    parser.add_argument(
        "--temperatura",
        type=float,
        default=0.8,
        help="Temperatura para geração (default: 0.8)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Máximo de tokens por resposta (default: 4096)",
    )
    parser.add_argument(
        "--categoria",
        choices=list(CATEGORIAS.keys()),
        help="Forçar uma categoria específica (default: rotação)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Mostra os prompts enviados à LLM"
    )
    args = parser.parse_args()

    base_prompt = load_base_prompt()
    existing_inputs = load_existing_inputs()
    initial_count = count_lines()

    # Preparar rotação de categorias
    cat_keys = list(CATEGORIAS.keys())
    random.shuffle(cat_keys)

    print(f"📄 Prompt base: {PROMPT_FILE.name}")
    print(f"📊 Dataset atual: {initial_count} entradas")
    print(f"🔗 Servidor: {args.url}")
    print(f"🔥 Temperatura: {args.temperatura}")
    print(f"📦 Batches: {args.batches}")
    if args.categoria:
        print(f"🎯 Categoria fixa: {CATEGORIAS[args.categoria]['nome']}")
    print()

    total_added = 0
    total_duplicates = 0
    total_similar = 0
    stats_por_categoria: dict[str, int] = {}

    for batch_num in range(1, args.batches + 1):
        # Escolher categoria
        if args.categoria:
            cat_key = args.categoria
        else:
            cat_key = cat_keys[(batch_num - 1) % len(cat_keys)]

        cat_nome = CATEGORIAS[cat_key]["nome"]
        print(f"━━━ Batch {batch_num}/{args.batches} — {cat_nome} ━━━")

        # Compor prompts
        system = build_system_prompt(base_prompt, cat_key)
        user = build_user_message(existing_inputs, cat_key)

        # Chamar LLM (com retry se resposta vazia)
        raw = ""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            print(
                f"  🤖 Gerando (tentativa {attempt}/{max_retries})..."
                if attempt > 1
                else "  🤖 Gerando..."
            )
            start = time.time()
            raw = call_llm(
                args.url, system, user, args.temperatura, args.max_tokens, args.verbose
            )
            elapsed = time.time() - start
            print(f"  ⏱ Resposta em {elapsed:.1f}s")

            if raw and raw.strip():
                break
            print("  ⚠ Resposta vazia da LLM, tentando novamente...")
            time.sleep(5)

        # Parsear
        entries = parse_jsonl_lines(raw)
        print(f"  📝 {len(entries)} linhas JSONL válidas")

        # Salvar resposta bruta para análise
        log_raw_response(batch_num, cat_key, raw, len(entries))

        # Filtrar duplicatas e similares
        new_entries = []
        for entry in entries:
            inp = entry.get("input", "").strip()
            if not inp:
                continue

            # Exact match
            if inp.lower() in {e.lower() for e in existing_inputs}:
                total_duplicates += 1
                print(f"  🔄 Duplicata exata ignorada: {inp[:50]}...")
                continue

            # Fuzzy match
            if is_similar(inp, existing_inputs, SIMILARITY_THRESHOLD):
                total_similar += 1
                print(f"  🔍 Similar demais ignorada: {inp[:50]}...")
                continue

            existing_inputs.append(inp)
            new_entries.append(entry)

        if not new_entries:
            print("  ⚠ Nenhuma entrada inédita neste batch")
        else:
            added = append_to_dataset(new_entries)
            total_added += added
            stats_por_categoria[cat_nome] = stats_por_categoria.get(cat_nome, 0) + added
            print(f"  ✅ {added} entradas adicionadas")

        if batch_num < args.batches:
            print()
            time.sleep(2)

    # Resumo final
    final_count = count_lines()
    print()
    print("━━━ Resultado Final ━━━")
    print(f"  📊 Antes: {initial_count} → Agora: {final_count}")
    print(f"  ➕ Adicionadas: {total_added}")
    print(f"  🔄 Duplicatas exatas: {total_duplicates}")
    print(f"  🔍 Similares rejeitadas: {total_similar}")
    if stats_por_categoria:
        print("  📂 Por categoria:")
        for cat, count in sorted(stats_por_categoria.items()):
            print(f"     • {cat}: {count}")


if __name__ == "__main__":
    main()
