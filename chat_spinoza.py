import torch
from unsloth import FastLanguageModel

# 1. Carregamento do Modelo Treinado
# Passamos a pasta onde salvamos os adaptadores LoRA ("spinoza_llama_lora").
# O Unsloth cuidará de carregar o modelo base (Llama 3.2-3B) e aplicar os nossos pesos de fine-tuning por cima.
max_seq_length = 2048

# AVISO: O caminho aqui deve apontar para a pasta onde você salvou o modelo (spinoza_llama_lora)
print("Carregando o modelo spinozista na RTX 4090...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="spinoza_llama_lora", 
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# Colocamos o modelo em modo rápido de inferência (desliga as funções de treinamento para economizar VRAM e ganhar velocidade)
FastLanguageModel.for_inference(model)

# 2. Template do Prompt
# Precisamos usar o MESMO template Alpaca que usamos no treinamento, senão o modelo não entende a pergunta.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Output:
{}"""

def perguntar_spinoza(dilema: str):
    # Formatamos a entrada do usuário
    entrada_formatada = alpaca_prompt.format(
        "Analise a situação cotidiana usando a lógica estrutural de Spinoza (afecções, conatus e razão) e ofereça um conselho.", 
        dilema, 
        "" # O output fica vazio para o modelo preencher
    )
    
    # Tokenizamos e enviamos para a GPU
    inputs = tokenizer([entrada_formatada], return_tensors="pt").to("cuda")
    
    # Geramos a resposta
    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True, temperature=0.7)
    
    # Decodificamos a resposta (tirando a parte do prompt que nós mesmos enviamos)
    resposta_completa = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extraímos apenas o que vem depois de "### Output:\n"
    resposta_limpa = resposta_completa.split("### Output:\n")[-1]
    return resposta_limpa.strip()

# 3. Loop de Chat Interativo
print("\n" + "="*50)
print(" Calculadora Ética Spinozista Inicializada!")
print(" Pressione Ctrl+C ou digite 'sair' para encerrar.")
print("="*50 + "\n")

while True:
    try:
        dilema_usuario = input("\n[Seu Dilema Moderno]: ")
        if dilema_usuario.lower() in ['sair', 'exit', 'quit']:
            break
            
        print("\n[Spinoza Pensa...]")
        conselho = perguntar_spinoza(dilema_usuario)
        
        print(f"\n[Spinoza Responde]:\n{conselho}")
        print("-" * 50)
        
    except KeyboardInterrupt:
        print("\nEncerrando a sessão filosófica.")
        break
