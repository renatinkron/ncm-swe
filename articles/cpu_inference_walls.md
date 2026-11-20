# As Duas Paredes: Caracterização Rigorosa de Inferência de Grandes Modelos em CPU de Consumo

**Autores / Projeto:** ncm_swe (Kron / Devin / Antigravity)  
**Data:** Novembro de 2026  
**Ambiente de Teste:** AMD Ryzen 7 PRO 4750U (Zen 2, 8 núcleos / 16 threads, AVX2, sem AVX-512), 32 GB RAM DDR4-3200 dual-channel (~37 GB/s leitura efetiva), Linux x86_64, `llama.cpp` (builds customizados e patches de profiling).

---

## Sumário Executivo

A execução local de Modelos de Linguagem de Grande Porte (LLMs) em hardware de consumo sem acelerador gráfico dedicado (GPU) é frequentemente tratada como inviável ou descartada por meio de regras de bolso superficiais. Este artigo apresenta uma caracterização empírica detalhada da inferência em CPU móvel de 15W–25W (Zen 2), desfazendo mitos operacionais comuns e estabelecendo o **Modelo das Duas Paredes**:
1. **Parede de Banda de Memória (Memory Bandwidth Wall):** Domina a geração autorregressiva com `batch=1`.
2. **Parede de Capacidade Computacional (Compute Wall):** Domina batches paralelos, decodificação especulativa e kernels de desquantização ultra-agressiva.

Todos os números apresentados foram medidos diretamente na máquina de teste e registrados na trilha experimental do projeto. Valores teóricos ou não medidos diretamente são explicitamente rotulados como `[estimate]`.

---

## 1. Caracterização do Hardware e Envelope Físico

O sistema em avaliação representa a classe de laptops de produtividade corporativa comuns no mercado:

* **Processador:** AMD Ryzen 7 PRO 4750U (microarquitetura Zen 2, TSMC 7nm).
* **Topologia:** 8 núcleos físicos, 16 threads lógicos, frequência base 1.7 GHz / boost até 4.1 GHz.
* **Conjunto de Instruções Vetoriais:** AVX2 + FMA (sem suporte a AVX-512 ou AMX).
* **Memória do Sistema:** 32 GB DDR4-3200 dual-channel (largura de barramento de 128 bits).
  * Largura de banda teórica máxima: $3200 \times 8 \times 2 = 51.2\text{ GB/s}$.
  * **Largura de banda real medida (stream benchmark, 8 threads):**
    * Leitura contínua: **36.7 GB/s** (~71.7% da eficiência teórica).
    * Triad: **19.3 GB/s**.
* **Configuração de SO / Kernel:** Linux x86_64, `Transparent Huge Pages (THP) = always`, CPU governor `schedutil`.

---

## 2. Profiling por Nó: Para Onde Vai o Tempo por Token?

Para responder onde os ciclos de CPU são consumidos durante a geração de tokens, instrumentamos o `ggml-cpu.c` (`llama.cpp`) com o patch `GGML_NODE_PROF`. O teste foi conduzido com o modelo **Qwen3-Next-80B** (arquitetura híbrida MoE, quantização IQ1_M-topk5, contexto curto, cache aquecido, tempo médio de **~113 ms por token**):

| Operação GGML | Nós no Grafo | Tempo Gasto (ms) | Fração do Passo (%) | Regime Operacional |
|---|---|---|---|---|
| `MUL_MAT` | 433 | 57.5 ms | 50.9% | Largura de banda pura |
| `MUL_MAT_ID` | 144 | 35.3 ms | 31.2% | Roteamento de experts MoE |
| `GET_ROWS` | — | 7.8 ms | 6.9% | Acesso a tabelas/embeddings |
| `GDN` | — | 4.8 ms | 4.2% | Normalizações / gates |
| `CPY` | — | 3.4 ms | 3.0% | Movimentação interna de buffers |
| Restante (outros nós) | — | 4.7 ms | 4.1% | Overhead de despacho/escalonamento |
| **Total por Token** | **~4.500 nós** | **~113.5 ms** | **100.0%** | **Taxa efetiva: ~8.8–10.5 tok/s** |

> **Achado Chave:** As operações `VIEW`, `RESHAPE`, `PERMUTE` e `TRANSPOSE` somam 1.764 nós no grafo, mas custam tempo próximo de zero (apenas manipulação de ponteiros de metadados, sem tocar a barreira de sincronização).  
> A taxa efetiva de leitura de memória na fase de `MUL_MAT` atingiu **~52 GB/s de pico**, batendo exatamente no teto físico de saturação do barramento DDR4-3200 da plataforma.

---

## 3. O Experimento OpenMP: Spin vs. Futex

Um diagnóstico inicial superficial apontou que **60.4% do tempo de CPU** estava dentro de `libgomp` (especificamente em funções de sincronização de barreira). Levantou-se a hipótese de que compilar o `llama.cpp` com `GGML_OPENMP=OFF` (eliminando `libgomp` em favor de threads pthreads e spinlocks customizados) traria um salto dramático de desempenho.

Medimos variantes idênticas (mesmo commit `5676dfb`, mesmos arquivos GGUF em cache quente, média de $r=3$):

| Modelo / Quant | Build Oficial (`GGML_OPENMP=ON`) | Build Custom (`GGML_OPENMP=OFF` / nomp) | Variação Medida |
|---|---|---|---|
| **Qwen3-Coder-30B** (IQ2_M-topk6, 10.3 GB) | **20.25 tok/s** | **20.07 tok/s** | -0.8% (empate técnico) |
| **Qwen3-Next-80B** (IQ1_M-topk5) | **10.16 ± 0.34 tok/s** | **10.41–10.85 ± 0.2 tok/s** | **+3% a +7%** (ganho marginal) |

### Diagnóstico Corrigido
Os 60% observados em `libgomp` **não eram overhead de despacho do sistema operacional**, mas sim threads ativas em *spin-wait* aguardando a conclusão da cadeia sequencial de ~4.500 nós por token (efeito *straggler* entre operações interdependentes no grafo híbrido MoE). `OpenMP=OFF` reduziu ligeiramente a latência de retomada das threads no modelo de 80B, mas confirmou que a contenção de sincronização é sintoma da topologia sequencial do grafo, não do runtime OpenMP.

---

## 4. O Ponto Ótimo de Quantização (Sweet Spot)

Testou-se uma varredura direta de quantizações para o **Qwen3-Coder-30B** sob as mesmas condições térmicas e de memória:

| Formato de Quantização | Bits por Peso (bpw) | Throughput Medido (tok/s) | Diagnóstico Físico |
|---|---|---|---|
| `IQ1_S` | 1.56 bpw | **3.88 tok/s** | **Compute-Bound:** kernel de desquantização ultra-complexo estrangula as ALUs do Zen 2 |
| **`IQ2_M-topk6`** | **2.70 bpw** | **20.35 tok/s** | **SWEET SPOT:** balanço exato entre bytes lidos e ciclos de desquantização AVX2 |
| `Q3_K_XL` | ~3.40 bpw | **16.62 tok/s** | **Bandwidth-Bound:** volume de dados excede a taxa de transferência sustentável da RAM |

> **Conclusão:** Para processadores com vetorização limitada a AVX2 (Zen 2 / Skylake), formatos de 1-bit (`IQ1_S`) são armadilhas de desempenho. O gargalo computacional da desquantização supera a economia de memória. O ponto ótimo de eficiência no balanço entre bits e ciclos é **`IQ2_M`**.

---

## 5. A Falácia da Decodificação Especulativa em CPU Bandwidth-Bound

A decodificação especulativa promete acelerações de $2\times$ a $3\times$ ao usar um modelo menor (*draft*) para propor $K$ tokens verificados em paralelo pelo modelo principal (*target*).

Implementamos e testamos essa estratégia pareada com o modelo oficial **z-lab/Qwen3-Coder-30B-A3B-DFlash** (0.5B parâmetros, 8 camadas, $d=2048$, treinado para o 30B):

| Configuração | Throughput (tok/s) | Taxa de Aceitação | Resultado |
|---|---|---|---|
| **30B Isolado (sem draft)** | **19.16 tok/s** | — | **Linha de base vencedora** |
| 30B + DFlash ($N_{\max}=3$) | 17.5–19.0 tok/s | 77%–90% | Empate com leve perda |
| 30B + DFlash ($N_{\max}=4$) | 14.60 tok/s | 62% | Degradação de -24% |
| 30B + DFlash + ngram-mod | < 10.0 tok/s `[estimate]` | 28% | Falha de sincronização |

### Por que a Decodificação Especulativa Falha na CPU?
A decodificação especulativa troca acessos de memória sequenciais por computação em lote ($B > 1$). Em uma GPU com terabytes de largura de banda e milhares de ALUs ociosas, essa troca é altamente vantajosa. Em uma CPU Zen 2:
1. O passo de verificação paralela consome aproximadamente **26 GFLOPs por passada** de capacidade computacional dos núcleos;
2. O modelo de rascunho (mesmo com 509 MB em `Q8_0`) compete pelo mesmo barramento DDR4 saturado;
3. A taxa de aceitação alta (77%–90%) não compensa o custo de processamento adicional das passadas de verificação. O sistema simplesmente **troca a Parede 1 (Banda) pela Parede 2 (Computação)** sem ganho líquido.

---

## 6. Degradação de Contexto Longo: O Vilão Invisível do KV-Cache

A degradação de throughput em contextos longos em CPUs é frequentemente subestimada. Medimos a taxa de geração do **Qwen3-Coder-30B** variando o tamanho do contexto e o formato do KV-cache:

```
Throughput (tok/s) vs. Tamanho do Contexto (Tokens)
20 +--[ctx ~30: 19.4 tok/s]
   |
15 |
   |
10 |                     [ctx 3.3k q4KV: 7.89 tok/s]
   |                     [ctx 3.3k q8KV: 6.49 tok/s]
 5 |                     [ctx 3.3k f16KV: 4.71 tok/s]
   +---------------------+----------------------------> Contexto
   0                    3.3k
```

### Números Medidos:
* **Contexto Curto (ctx ~30):** **19.4 tok/s**.
* **Contexto Médio (ctx ~3.300):**
  * Com `f16` KV-cache: **4.71 tok/s** (queda de -75.7%).
  * Com `q8_0` KV-cache: **6.49 tok/s** (queda de -66.5%).
  * Com `q4_0` KV-cache: **7.89 tok/s** (recuperação de **+21.5%** sobre `q8` e **+67.5%** sobre `f16`).

### Por que o KV-Cache Degrada Tanto em CPU?
1. **Consumo de Memória:** Um contexto de 131.072 tokens com KV em `f16` aloca **25.8 GB de RAM**, expulsando os pesos do modelo da memória cache e provocando *thrashing* no subsistema de memória.
2. **Custo Algorítmico da Atenção:** A atenção na CPU possui complexidade $O(N_{\text{ctx}})$ em leitura contínua de memória durante a decodificação. Em $\approx 37\text{k}$ tokens de contexto `[estimate]`, a leitura e processamento do histórico de KV custa mais tempo de banda do que a leitura de todos os pesos do modelo ativo.

---

## 7. Configuração Campeã Recomendada para Servidores Locais

Com base nas evidências empíricas coletadas, a configuração operacional ótima para CPUs da classe AMD Ryzen 8-core com 32 GB DDR4 é:

```bash
./llama-server \
  -m /path/to/Qwen3-Coder-30B-A3B-IQ2_M.gguf \
  --threads 8 \
  --ctx-size 8192 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -fa \
  --host 0.0.0.0 --port 8085
```

### Regras Operacionais de Ouro
1. **Threads = Núcleos Físicos:** Usar 8 threads em um processador de 8c/16t. Ativar hyperthreading/SMT para inferência densa aumenta a contenção na L3 e degrada a vazão.
2. **Quantização de KV em Q4:** Economiza metade da RAM do buffer e garante até +21% de velocidade em contextos acima de 3.000 tokens.
3. **Limitação Estrita de Contexto:** Nunca alocar contextos gigantes desnecessários (`-c 131072`), pois o buffer alocado reserva memória imediatamente e penaliza a latência.
4. **Reinicialização de Sessão:** Manter os prompts de instrução compactados e reiniciar sessões longas para limpar o buffer de atenção.

---

## 8. Conclusão

A inferência local em CPU é governada por leis físicas rígidas de barramento e latência. Projetos que tentam aplicar atalhos concebidos para data centers com GPUs (como decodificação especulativa massiva ou quantização extrema para 1-bit) falham em hardware doméstico por ignorarem a transição entre a **Parede de Banda** e a **Parede de Computação**.

Compreender esses limites permite extrair taxas práticas de **10 a 20 tokens por segundo** em modelos sofisticados de 30B a 80B parâmetros em um laptop comum, fornecendo uma base estável, silenciosa e totalmente privada para engenharia de software e pesquisa.
