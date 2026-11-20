import json, urllib.request, ast, re, time, random
random.seed(3)
INTENTS = [
 # domínios descobertos + os que FALHARAM no eval (arquivo, binária, média, sort, pilha)
 "le um arquivo texto e retorna as linhas", "escreve linhas num arquivo",
 "le um csv e retorna lista de dicts", "conta linhas nao vazias de um arquivo",
 "appenda uma linha no fim do arquivo", "le json com fallback se invalido",
 "busca binaria em lista ordenada", "busca linear retornando indice ou -1",
 "ordena lista de dicts por chave", "merge sort recursivo", "quick sort",
 "insertion sort", "seleciona o k-esimo menor", "median of a list",
 "moving average com janela k", "exponential moving average",
 "stack com push pop peek", "queue com enqueue dequeue", "deque circular",
 "linked list simples", "min-heap com push e pop", "priority queue",
 "classe de conta bancaria com deposito", "classe retangulo com area",
 "classe ponto com distancia", "classe carrinho de compras",
 "dataclass de produto", "classe com __repr__ customizado",
 "property com getter e setter", "classe contador de ocorrencias",
 "decorator de timing", "decorator de retry", "decorator memoize",
 "decorator que valida argumentos", "context manager de timer",
 "context manager de arquivo temp", "generator de fibonacci",
 "generator que le arquivo por blocos", "generator de pares (i, v)",
 "iterator customizado de range reverso", "pipeline de generators",
 "async fetch de uma url com aiohttp", "async gather de varias tasks",
 "async funcao com timeout", "async retry com backoff",
 "regex para validar email", "regex para extrair numeros",
 "regex para slugificar", "split por multiplos delimitadores",
 "parse de data iso", "diferenca em dias entre datas",
 "proximo dia util", "timestamp para string legivel",
 "dict aninhado com get seguro", "inverte um dict", "merge profundo de dicts",
 "filtra dict por predicado", "agrupa lista por chave", "conta com Counter",
 "flatten de lista aninhada", "chunks de tamanho n", "rotaciona lista",
 "dedup preservando ordem", "intersecao de listas", "diferenca simetrica",
 "particiona por predicado", "tira amostra aleatoria",
 "mapeia com indice", "transpoe matriz", "soma colunas de matriz",
 "fatorial iterativo", "fibonacci com memoizacao", "gcd e lcm",
 "primo check com trial division", "sieve de eratostenes", "potencia modular",
 "converte para binario", "parse de float seguro", "clamp valor em range",
 "mapeamento run-length encode", "run-length decode", "string to camel",
 "snake_to_camel e camel_to_snake", "trunca string com reticencias",
 "centra string com padding", "wrap texto em largura n", "plural simples",
 "remove acentos de string", "conta vogais", "titulo capitalizado",
 "try except com log de erro", "excecao customizada", "valida tipo com raise",
 "safe division com default", "parse de args simples", "env var com default",
 "timer com time.time", "memoization com dict", "cache lru simples",
 "debounce de funcao", "throttle de chamadas", "rate limiter simples",
 "compreensao de lista aninhada", "dict comprehension com filtro",
 "set comprehension unico", "generator expression soma",
 "lambda como key de sort", "any all combinados", "enumerate com start",
 "zip com strict", "itertools.chain", "itertools.groupby",
 "compara duas listas ignorando ordem", "deep equality de dicts",
 "hash de arquivo md5", "uuid4 string", "token aleatorio seguro",
 "base64 encode decode", "url encode de params", "json pretty print",
 "dataclass ordenada", "enum de estados", "singleton por classe",
 "observer pattern simples", "publisher subscriber minimo",
 "adapter de interface", "factory por string", "strategy pattern",
 "template method", "builder de query", "wrapper com functools.wraps",
 "threadsafe counter com lock", "queue entre threads", "worker simples",
 "chamada subprocess com captura", "le env vars com prefixo",
 "path join seguro", "lista arquivos por extensao", "tamanho de diretorio",
 "resolve path relativo", "tempdir com cleanup", "walk diretorio",
 "le stdin linha a linha", "print formatado tabela", "barra de progresso",
 "colore texto com ansi", "menu interativo simples", "prompt com default",
]
random.shuffle(INTENTS)

def ask(intent):
    req = urllib.request.Request(
        "http://127.0.0.1:8085/v1/chat/completions",
        data=json.dumps({
            "messages": [
                {"role": "system", "content": "Output ONLY a single short Python function or class. No markdown, no docstrings, no comments inside code, no tests, no prose. Max 14 lines."},
                {"role": "user", "content": f"Write Python code: {intent}"}],
            "max_tokens": 380, "temperature": 0.75, "top_p": 0.9}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        return d["choices"][0]["message"]["content"]
    except Exception:
        return None

def clean(code):
    code = re.sub(r"^```(?:python)?\n?|```$", "", code.strip(), flags=re.M)
    lines = [l for l in code.split("\n")
             if not re.match(r"^\s*(here|this|output|note|explanation)", l, re.I)]
    return "\n".join(lines).strip()

kept, fails = [], []
t0 = time.time()
for i, it in enumerate(INTENTS):
    code = ask(it)
    if code is None:
        fails.append((it, "api")); continue
    unit = f"# {it}\n{clean(code)}"
    try:
        ast.parse(unit)
        kept.append(unit)
    except SyntaxError:
        fails.append((it, "syntax"))
    if (i+1) % 20 == 0:
        print(f"{i+1}/{len(INTENTS)} kept={len(kept)} "
              f"({(time.time()-t0)/(i+1):.1f}s/u)", flush=True)

out = "/work/artifacts/ncm_code" if __import__('os').path.isdir("/work/artifacts") else "/tmp"
open(f"{out}/qwen_units_raw.json", "w").write(json.dumps(kept, indent=1))
open(f"{out}/qwen_units_fails.json", "w").write(json.dumps(fails, indent=1))
txt = "\n\n".join(kept)
open(f"{out}/qwen_units.txt", "w").write(txt)
print(f"\nDONE kept={len(kept)}/{len(INTENTS)} fails={len(fails)} "
      f"elapsed={(time.time()-t0)/60:.0f}min → {out}/qwen_units.txt")
