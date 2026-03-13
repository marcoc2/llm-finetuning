# Calculadora Ética Spinozista

Este repositório contém os scripts e o dataset base para realizar o treinamento (_fine-tuning_) de um modelo de inteligência artificial Llama 3.2 utilizando a filosofia sistemática de Baruch Spinoza. O objetivo é criar um assistente filosófico capaz de analisar dilemas cotidianos através da estrutura lógica da _Ética_ (afecções, conatus, razão, imaginação).

O treinamento é feito usando a biblioteca [Unsloth](https://github.com/unslothai/unsloth), que provê uma velocidade extrema e otimização pesada no consumo de VRAM das placas de vídeo.

## Estrutura do Projeto

- `dataset_spinoza.jsonl`: Dataset base com dilemas sintéticos e conselhos filosóficos formatados para instrução.
- `spinoza_finetune.py`: Script de _fine-tuning_ do modelo `Llama-3.2-3B` utilizando o Unsloth e PEFT (LoRA).
- `chat_spinoza.py`: Script interativo final. Ele carrega o seu modelo já treinado no console para você bater papo e pedir conselhos para a máquina.
- `prompt_geracao_spinoza.txt`: Um _prompt_ cuidadosamente esculpido a partir da lógica de formalização matemática (Coq) para você gerar mais material sintético caso deseje expandir o dataset no futuro usando outras LLMs maiores.
- `fix_json.py`: Um canivete suíço em Python para corrigir possíveis aspas falhas (erros de _backslash_) que LLMs de terceiros possam cometer ao gerar novos dados JSONL para você.

## Requisitos de Hardware

Para realizar o _fine-tuning_ local do Llama 3.2-3B, recomenda-se fortemente o uso de uma **GPU da NVIDIA** com suporte a arquitetura moderna (Ampere, Ada, etc), como a série RTX 3000 ou 4000, possuindo pelo menos **8GB de VRAM**. (Nos testes, foi perfeitamente acomodado em uma RTX 4060 ou RTX 4090).

---

## 🐧 Instalação no Ubuntu (Linha Debian/Linux)

Aqui estão os passos limpos para compor seu ambiente Linux e não ter conflitos:

**1. Instale ferramentas básicas**

```bash
sudo apt update
sudo apt install build-essential python3.10 python3.10-venv python3-pip git
```

**2. Crie e ative um Ambiente Virtual Isolado**

```bash
python3.10 -m venv venv_spinoza
source venv_spinoza/bin/activate
```

**3. Instale o PyTorch (focado na versão de CUDA 12.1+ que estiver no seu sistema)**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**4. Instale o Unsloth e dependências de Otimização**

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets pandas pyarrow
```

> **Aviso de Troubleshooting:** Em distribuições recentes do Ubuntu e usando CUDA mais novo, o PyTorch pode não encontrar facilmente a biblioteca `libcusparseLt.so.0`. Caso ganhe este erro ao iniciar o Python, injete a variável de ambiente abaixo ou adicione-a no script `activate` do seu venv:
> `export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH`

---

## 🪟 Instalação no Windows

O projeto depende de compilações profundas relacionadas a CUDA (via Unsloth) que são altamente instáveis no Windows nativo. Portanto, o caminho suportado pelo ecossistema IA atualmente é o WSL2.

### Método 1: Recomendado Oficialmente (Ubuntu no Windows via WSL2)

Este método vai rodar um Linux por debaixo dos panos com acesso à sua GPU nativamente.

**1. Instale o WSL2 via PowerShell (como Administrador)**

```powershell
wsl --install
```

_Se você já possuía o WSL instalado, abra a distribuição (ex: Ubuntu-22.04)._

**2. Verifique o driver NVIDIA**
Em sua máquina Windows nativa certifique-se de que instalou o Driver NVIDIA mais recente. (Você **NÃO DEVE** tentar instalar drivers de tela no sistema interno do WSL. O WSL puxa as permissões de GPU por padrão direto do Windows).
Dentro do WSL, digite `nvidia-smi` para confirmar o funcionamento.

**3. Siga o Guia Linux**
Tendo o terminal do Ubuntu aberto em sua máquina Windows, vá para a etapa de "Instalação no Ubuntu" descrita no tópico acima e faça todos os `apt` e `pip install` descritos! Tudo funcionará sem barreiras.

### Método 2: Nativo pelo Windows (Módulo Experimental)

Caso recuse o WSL, instale no Windows por sua própria conta e risco, seguindo o passo a passo:

**1.** Tenha baixado o [Python 3.10](https://www.python.org/downloads/release/python-31011/) marcando "Add to PATH" ao instalar. E tenha instalado também [Git para Windows](https://git-scm.com/download/win).
**2.** Abra uma janela do PowerShell e crie um Ambiente Virtual:

```powershell
python -m venv venv_spinoza
.\venv_spinoza\Scripts\activate
```

**3.** Instale um PyTorch para sua NVIDIA compatível no Windows:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**4.** Instale as livrarias de AI em versão Windows puro:

```powershell
pip install unsloth "trl<0.9.0" peft accelerate datasets pandas pyarrow
# Para o uso de memórias 4-bit e 8-bit no windows puro, recomenda-se bibliotecas específicas de C++ como o `bitsandbytes-windows`.
python -m pip install bitsandbytes==0.41.1 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

---

## 🎬 Como Rodar a Aplicação

**Etapa Inicial: Treinamento**
Com tudo configurado e sua `venv` ativa, inicie a lapidação do mestre. Este processo fará o Download do Llama-3.2 da nuvem e adaptará suas sinapses ao _dataset_ Spinozista.

```bash
python spinoza_finetune.py
```

> _Ao fim, você verá uma pasta `spinoza_llama_lora/` contendo os resultados._

**Etapa Final: Utilização do Chat**
Tendo a pasta do seu treinamento concluída em mãos (ou tendo baixado da nuvem de volta), você pode abrir o console e conversar sobre os seus problemas da vida para ele analisar de acordo com a razão, os vetores de emoções tristes e alegres:

```bash
python chat_spinoza.py
```

_(Para parar a conversa: pressione `Ctrl+C` ou digite 'sair')_
