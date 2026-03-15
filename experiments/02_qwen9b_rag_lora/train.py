import torch
import argparse
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
# Vamos usar o Qwen3.5-9B em 4 bits (QLoRA) para caber na RTX 4090.
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-9B",  # Conforme solicitado pelo usuário
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,  # Auto-detect (usará bf16 na 4090)
)

# 2. Configuração dos Adaptadores LoRA
# Treinamos os módulos específicos (Qwen usa arquitetura Llama-like, então os target modules são os mesmos)
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
    use_gradient_checkpointing="unsloth",  # Otimização de memória do Unsloth
    random_state=3407,
)

# 3. Preparação do Dataset
# O RAG trará o conteúdo depois. O LoRA precisa apenas aprender o ESTILO e FORMATO spinozista.
# Mapeamos o dataset sintético usando um formato de ChatML compatível com Qwen.
# Mas como estamos treinando a formatação (e temos instruction/input/output), vamos usar o formato Alpaca
# (ou o mesmo que funcionou no exp 01) e ensinar o Qwen a responder como nele.
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
        # Must add EOS_TOKEN, otherwise generation will go on forever
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Pegar argumentos para dry-run
parser = argparse.ArgumentParser()
parser.add_argument(
    "--max_steps", type=int, default=120, help="Número de steps. Use 1 para dry run."
)
args = parser.parse_args()

# 4. Configuração do SFTTrainer
# Ajustamos max_steps baseado no dataset um pouco maior, ou podemos usar epochs.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=args.max_steps,  # 120 steps default para o Qwen pegar o formato
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

# 5. Treinamento
print("Iniciando o treinamento do LoRA de estilo Spinozista no Qwen 3.5-9B...")
trainer_stats = trainer.train()

# 6. Salvar Adapters
print(f"Treinamento concluído. Salvando adaptadores em {ADAPTER_DIR}...")
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

# 7. Teste Rápido de Inferência (Apenas LoRA, sem RAG ainda)
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Aja como um spinozista. Analise a situação usando os conceitos de afeto e conatus de acordo com Spinoza.",
            "Estou sentindo inveja do sucesso do meu colega.",
            "",
        )
    ],
    return_tensors="pt",
).to("cuda")

print("\n=== Resposta de Teste (Apenas LoRA -> Tom) ===")
outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
print(tokenizer.batch_decode(outputs)[0])
print("==============================================")
