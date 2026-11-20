"""Teacher v3 — canal intenção→código: '# <intent>' antes do unit.
Variável única vs v2: só adiciona o comentário de intenção (PT/EN ~50/50);
~20% ficam sem comentário p/ não viciar o canal bare-def.
Importa as famílias do _teacher_gen (mesma semântica, +comment)."""
import ast, json, os, random, sys
sys.path.insert(0, "/repo/scripts")
import _teacher_gen as TG  # noqa: E402

OUTD = "/work/artifacts/ncm_code"
SEED = int(os.environ.get("SEED", "21"))
N_OUT = int(os.environ.get("N_OUT", "4000"))
OUT_NAME = os.environ.get("OUT_NAME", "teacher_synth_v3.txt")
P_BARE = 0.20  # fração sem comentário de intenção

INTENTS = {
    "fam_accumulate": ["soma os valores de {n}", "accumulate the values",
                       "totaliza os itens", "sum every entry"],
    "fam_transform": ["transforma cada item", "map each element",
                      "aplica a transformação"],
    "fam_filter": ["filtra os itens válidos", "keep only matching items",
                   "remove os inválidos"],
    "fam_validator": ["verifica se os dados são válidos",
                      "check whether the input is valid"],
    "fam_lookup": ["busca com valor padrão", "look up with fallback",
                   "acha a chave ou devolve default"],
    "fam_guard": ["protege contra entrada vazia", "guard against bad input"],
    "fam_string": ["normaliza o texto", "clean up the string"],
    "fam_math": ["calcula o valor", "compute the result"],
    "fam_class": ["estrutura simples de dados", "small data container"],
    "fam_recursive": ["soma recursivamente na estrutura",
                      "recursive total over the tree"],
    "fam_loop_build": ["constrói a sequência", "build the sequence"],
    "fam_exception": ["converte com segurança", "convert safely with default"],
    "fam_dict_merge": ["junta os dois dicionários", "merge two dicts"],
    "fam_generator": ["produz valores sob demanda", "yield values lazily"],
    "fam_zip_unpack": ["pareia chaves e valores",
                      "zip keys and values together"],
    "fam_nested_get": ["busca em dicionário aninhado",
                      "safely get a nested value"],
    "fam_async": ["executa assíncrono", "run asynchronously",
                  "versão assíncrona da coleta"],
    "fam_method_class": ["cache simples com get e put",
                         "keyed store with get and put"],
    "fam_decorated": ["versão com decorador", "decorated variant"],
    "fam_with_ctx": ["lê ou escreve o arquivo", "read or write the file"],
    "fam_while": ["consome a fila sem repetir",
                  "drain the queue visiting each once"],
    "fam_comprehension": ["mapeia os dados em uma linha", "one-line mapping"],
    "fam_lambda": ["ordena pela chave", "sort by key"],
    "fam_import": ["normaliza o caminho", "normalize the path"],
    "fam_typed": ["converte respeitando o tipo", "convert honoring the type"],
    "fam_docstring": ["resume os dados", "summarize the inputs"],
}


def intent_for(fam_name, src):
    pool = INTENTS.get(fam_name, ["processa os dados", "process the data"])
    it = random.choice(pool)
    if "{n}" in it:
        it = it.format(n=random.choice(TG.NOUNS))
    return f"# {it}"


def main():
    random.seed(SEED)
    seen, out = set(), []
    stats = {"target": N_OUT, "families": len(TG.FAMS), "tries": 0,
             "parse_fail": 0, "dup": 0, "bare": 0, "cond": 0}
    while len(out) < N_OUT and stats["tries"] < N_OUT * 25:
        stats["tries"] += 1
        fam = random.choice(TG.FAMS)
        src = fam()
        if "\n\n" in src:
            stats["parse_fail"] += 1
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            stats["parse_fail"] += 1
            continue
        if not any(not (isinstance(n, ast.Expr) and
                        isinstance(n.value, ast.Constant))
                   for n in tree.body):
            continue
        if random.random() < P_BARE:
            stats["bare"] += 1
        else:
            src = intent_for(fam.__name__, src) + "\n" + src
            stats["cond"] += 1
        if src in seen:
            stats["dup"] += 1
            continue
        seen.add(src)
        out.append(src)
    open(OUTD + "/" + OUT_NAME, "w").write("\n\n".join(out))
    stats["kept"] = len(out)
    json.dump(stats, open(OUTD + "/" +
                          OUT_NAME.replace(".txt", "_stats.json"), "w"),
              indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
