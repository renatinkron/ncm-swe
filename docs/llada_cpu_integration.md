# Dossiê Técnico: Integração de Modelos de Difusão de Texto (LLaDA / Dream) em CPU

**Autor:** Antigravity (Pesquisa & Arquitetura)  
**Destinatário:** Devin / Mantenedores do projeto `ncm_swe`  
**Data:** 20 de Novembro de 2026  
**Status de Rigor:** Todas as fontes verificadas contra repositórios públicos e papers. Gaps de maturidade marcados explicitamente.

---

## 1. Verificação do Motor: `diffuse-cpp`

* **Repositório GitHub:** [iafiscal1212/diffuse-cpp](https://github.com/iafiscal1212/diffuse-cpp)
* **Organização no Hugging Face:** [diffuse-cpp](https://huggingface.co/diffuse-cpp)
* **Base Tecnológica:** Construído diretamente sobre a biblioteca **GGML** (a mesma fundação C/C++ do `llama.cpp`).
* **Formato de Pesos:** GGUF nativo (`Q4_K_M`, `Q8_0`, `F16`).
* **Modelos Disponibilizados em GGUF:**
  1. `diffuse-cpp/LLaDA-8B-Instruct-GGUF`
  2. `diffuse-cpp/Dream-v0-Instruct-7B-GGUF`
* **Inovações de Performance:**
  * Implementa **Inter-Step KV Cache**: reaproveita ativações entre passos de difusão, recalculando apenas o conjunto ativo (tokens ainda mascarados e vizinhança imediata cuja distribuição variou).
  * Algoritmo de amostragem: *low-confidence remasking* e estratégia `entropy_exit` para encurtar passadas em tokens com alta certeza.
* **Nível de Maturidade (Honesto):** `[early-stage research / functional CLI]`
  * **O que funciona:** Compilação via CMake (`Release`), carregamento de tensores quantizados GGUF, alocação de threads CPU via GGML, execução do loop de desmascaramento por $S$ passos.
  * **Lacuna Operacional Atual:** O executável CLI (`diffuse-cli`) recebe sequências pré-tokenizadas como lista de inteiros (`--tokens "128000,3923,..."`). Ele **não** possui tokenizer BPE embutido em C++ como o `llama-cli`.
  * **Requisito para Integração:** Necessita de um wrapper Python leve (ex: via `tiktoken` ou Hugging Face `tokenizers`) para converter prompt de texto $\to$ IDs $\to$ chamar `diffuse-cli` $\to$ decodificar saída.

---

## 2. Landscape Não-Autorregressivo (2025–2026) em Hardware Doméstico (Ryzen 7 4750U / 32GB)

| Modelo | Pesos Abertos? | Formato / Runtime | Viabilidade no 4750U (32GB DDR4) | Avaliação Prática |
|---|---|---|---|---|
| **Dream-7B** (HKU / Huawei) | **SIM** (Apache 2.0) | GGUF (`diffuse-cpp`) ou PyTorch (`transformers`) | **ALTA** (consome ~4.5 GB em Q4_K_M) | **Recomendação #1:** Inicializado dos pesos do Qwen2.5-7B + CART noise scheduling. Excelente em código e lógica. |
| **LLaDA-8B** (GSAI-ML) | **SIM** (Apache 2.0) | GGUF (`diffuse-cpp`) ou PyTorch (`transformers`) | **ALTA** (consome ~5.2 GB em Q4_K_M) | **Recomendação #2:** Modelo seminal de difusão de 8B (2.3T tokens). Suporta edição não-monotônica nativa. |
| **DiffusionGemma** (Google) | **SIM** (Apache 2.0) | PyTorch / JAX | **BAIXA** (MoE 25.2B total, ~3.8B ativo) | Sem runtime C++ quantizado maduro; alocação em RAM pesada sem GGUF. |
| **Mercury** (Inception Labs) | **NÃO** (Proprietário) | API comercial fechada | **NULA** | Datacenter H100 only; sem pesos públicos. |
| **Gemini Diffusion** (DeepMind) | **NÃO** (Proprietário) | Pesquisa interna | **NULA** | Sem pesos públicos. |
| **MDLM / SEDD** (NeurIPS/ICML) | **SIM** (Acadêmico) | Scripts PyTorch de pesquisa | **MÉDIA** (apenas modelos sub-1B) | Modelos de prova de conceito acadêmica; superados por Dream e LLaDA. |

---

## 3. Protocolo Padrão de Avaliação Qualidade × Passos em dLLMs

Na literatura de modelos de difusão de texto (especialmente LLaDA arXiv:2502.09992 e Dream-7B 2025), a qualidade não é medida por perplexidade por token clássica, mas sim por:

### 3.1 Métrica Central de Trade-Off: Curva de Pareto NFE × Acurácia
* **NFE (Number of Function Evaluations):** Número de passadas completas (*forward passes*) pela rede. No caso autoregressivo, $\text{NFE} = L$ (comprimento da saída). Na difusão, $\text{NFE} = S$ (número de passos de amostragem configurados).
* **Grid de Passos Padrão:** Avalia-se a geração fixando $S \in \{16, 32, 64, 128, 256, 512\}$.
* **Benchmarks de Código:** **HumanEval (pass@1)** e **MBPP**.
* **Comportamento Observado na Literatura:**
  * Em $S \le 32$: Ocorre colapso sintático e alta taxa de erros léxicos sem amostragem guiada por entropia.
  * Em $S = 128 \to 256$: Ponto de inflexão onde modelos como Dream-7B e LLaDA atingem paridade com autoregressivos em tarefas estruturadas.
  * Em $S = 512$: Convergência de qualidade máxima (ponto ótimo para geração de código complexo).

---

## 4. Arquitetura Proposta: O "Painel Renderizador" sobre Base Pronta

O script `scripts/ncm_panel.py` já desenhou a interface ideal (amostras, preview por nível, controle de temperatura/CFG, filtros de saída). A transposição dessa interface para um backend dLLM (como LLaDA ou Dream via `diffuse-cpp`) é direta:

```
+-------------------------------------------------------------------+
|               scripts/ncm_panel.py (ou diffuse_panel.py)          |
|                                                                   |
|   --steps S       -> Passadas de refinamento (16, 32, 64, 256...) |
|   --preview       -> Exibir o buffer a cada K passos de denoising |
|   --remasking     -> Estratégia de filtro (low_conf, entropy_exit)|
|   --n / --seed    -> Amostras independentes (samples de render)   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               Wrapper de Tokenização Local (Python)               |
|            Prompt -> BPE Token IDs -> CSV string de IDs           |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|            diffuse-cli (C++ / GGML engine em CPU Zen 2)           |
|            -m llada-8b-q4km.gguf -s <steps> --remasking ...       |
+-------------------------------------------------------------------+
```

Isso viabiliza o conceito de "modelo-renderizador" com precisão e controle de qualidade ajustável em tempo real, sem necessidade de gastar semanas treinando um modelo do zero.
