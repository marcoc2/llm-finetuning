import torch
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "spinoza_etica.jsonl"
ADAPTER_DIR = SCRIPT_DIR / "adapter"

# 1. Configuração Inicial e Carregamento do Modelo
# O FastLanguageModel do Unsloth carrega o modelo de linguagem otimizado para maior velocidade e menor uso de memória.
# Utiliza-se 'load_in_4bit=True' para carregar os pesos em quantização de 4 bits, o que reduz drasticamente
# a VRAM necessária, permitindo rodar na RTX 4090 de forma folgada. O 'max_seq_length=2048' define o tamanho máximo
# de tokens de contexto que o modelo pode processar de uma vez durante o treinamento.
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# 2. Configuração dos Adaptadores LoRA
# LoRA (Low-Rank Adaptation) congela a maioria dos pesos originais do modelo e insere matrizes de baixo rank (adaptadores)
# que serão as únicas partes treinadas. O 'r=16' define o rank (tamanho/capacidade) dessa matriz e 'lora_alpha=16'
# é um fator de escala de aprendizado. Treinamos módulos específicos de atenção e feed-forward com o 'target_modules'.
# O dropout é zerado ('lora_dropout=0') por recomendação padrão de performance no Unsloth.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. Preparação do Dataset
# Carrega os dados de instrução do JSONL que deve seguir o padrão Alpaca (instruction, input, output).
# A formatação mapeia esse dataset transformando os dados estruturados em uma única string textual consolidada,
# que é a forma esperada pelo LLM para aprender a prever o próximo token e formatar sua resposta corretamente.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Output:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 4. Configuração do SFTTrainer
# O SFTTrainer constrói e padroniza o loop de treinamento. Configuramos um número fixo
# curto de passos ('max_steps=60') para uma prova de conceito rápida. O batch size ('per_device_train_batch_size=2')
# e a acumulação de gradientes controlam a memória consumida por step de otimização.
# Usa-se um otimizador eficiente de 8 bits ('adamw_8bit') e seleciona-se 'bf16' automaticamente
# caso o hardware suporte (uma RTX 4090 suporta e tem benefícios de estabilidade em bfloat16).
# O parâmetro 'logging_steps=10' garante exibição das métricas no console com frequência.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=str(SCRIPT_DIR / "outputs"),
    ),
)

# 5. Treinamento e Inferência Pós-Treino
# 'trainer.train()' começa a repetição de steps onde os pesos do LoRA são atualizados com retropropagação.
# Após concluir, 'model.save_pretrained' salva os adaptadores finos na pasta 'spinoza_llama_lora', economizando armazenamento.
# Depois o modelo é posto em modo de inferência local ('for_inference'), um prompt sobre Spinoza é formatado
# usando a métrica Alpaca e submetido ao modelo para geração ('model.generate()'), validando a absorção do estilo.
trainer_stats = trainer.train()

model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Aja como um spinozista. Analise a situação usando os conceitos de afeto e conatus de acordo com Spinoza.",  # instruction
            "Estou sentindo inveja do sucesso do meu colega.",  # input
            "",  # output (deixado em branco para completar com a geração)
        )
    ],
    return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
print("\n=== Resposta de Teste (Inferência) ===\n")
print(tokenizer.batch_decode(outputs)[0])
