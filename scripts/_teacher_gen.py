"""Sintetizador de teacher units: famílias de padrões reais de código,
compostos com nomes/constantes/operações randomizados. Toda unidade é
ast.parse-válida por construção + verificada. Volume programático com
correção garantida — o professor carrega o conhecimento nos templates."""
import ast, json, os, random

OUTD = "/work/artifacts/ncm_code"
SEED = int(os.environ.get("SEED", "7"))
N_OUT = int(os.environ.get("N_OUT", "4000"))
OUT_NAME = os.environ.get("OUT_NAME", "teacher_synth.txt")

NOUNS = ["items", "values", "nums", "data", "records", "entries", "seq",
         "xs", "elems", "rows"]
VERBS_GET = ["sum", "count", "collect", "gather", "find", "filter",
             "map", "select", "total", "aggregate"]
PROPS = ["empty", "sorted", "valid", "positive", "numeric", "even",
         "unique", "balanced", "clean", "uniform"]
TYPES = ["int", "str", "float", "list", "dict", "tuple"]
OPS = ["+", "-", "*", "//", "%", "**"]
CMP = [">", "<", ">=", "<=", "==", "!="]
NAMEPOOL = ["alpha", "beta", "gamma", "delta", "node", "user", "item",
            "score", "rate", "level", "token", "entry", "metric",
            "price", "count", "ratio", "index", "value", "key"]


def r(xs):
    return random.choice(xs)


def names(n):
    return random.sample(NAMEPOOL, min(n, len(NAMEPOOL)))


def fam_accumulate():
    n = r(NOUNS)
    init = r(["0", "0.0", "[]", '""', "1"])
    op = r(OPS[:5])
    cond = r(["", f"        if x {r(CMP)} {random.randint(0, 9)}:\n",
              f"        if isinstance(x, {r(TYPES)}):\n",
              f"        if x:\n"])
    body = f"acc {op}= x" if op != "*" else "acc *= x"
    ind = "            " if cond else "        "
    inner = f"{cond}{ind}{body}" if cond else f"        {body}"
    tail = 'return "".join(acc) if isinstance(acc, list) is False else acc' \
        if init == '""' else "return acc"
    tail = "return acc"
    return (f"def accumulate_{r(names(1))}({n}):\n"
            f"    acc = {init}\n    for x in {n}:\n{inner}\n    {tail}")


def fam_transform():
    n = r(NOUNS)
    expr = r([f"x {r(OPS[:4])} {random.randint(2, 9)}", "x * x",
              "abs(x)", "str(x)", "len(x)", "x.lower()", "x.strip()",
              f"x {r(CMP)} {random.randint(0, 9)}"])
    return f"def transform_{r(names(1))}({n}):\n    return [{expr} for x in {n}]"


def fam_filter():
    n = r(NOUNS)
    cond = r([f"x {r(CMP)} {random.randint(0, 20)}",
              "x is not None", "x", "len(x) > 0",
              f"isinstance(x, {r(TYPES)})"])
    return (f"def select_{r(names(1))}({n}):\n"
            f"    return [x for x in {n} if {cond}]")


def fam_validator():
    conds = [f"return len({r(NOUNS)}) {r(CMP)} {random.randint(0, 9)}",
             f"return all(x {r(CMP)} 0 for x in {r(NOUNS)})",
             f"return {r(NOUNS)} == sorted({r(NOUNS)})",
             f"return len(set({r(NOUNS)})) == len({r(NOUNS)})"]
    return f"def is_{r(PROPS)}(x):\n    {r(conds)}".replace("(x)", f"({r(NOUNS)})")


def fam_lookup():
    name = r(names(1))
    return (f"def index_{name}({r(NOUNS)}):\n"
            f"    out = {{}}\n"
            f"    for i, x in enumerate({r(NOUNS)}):\n"
            f"        out.setdefault(x, []).append(i)\n"
            f"    return out")


def fam_guard():
    name = r(names(1))
    checks = random.randint(1, 3)
    body = []
    for _ in range(checks):
        body.append(f"    if {r(NAMEPOOL)} is None:\n        return {r(['None', '0', 'False', '[]', '-1'])}")
    body.append(f"    return {r(NAMEPOOL)} {r(OPS[:4])} {random.randint(1, 9)}")
    params = ", ".join(names(2))
    return f"def safe_{name}({params}):\n" + "\n".join(body)


def fam_string():
    name = r(names(1))
    ops = [f"text.replace({r(['chr(32)', repr('_'), repr('-')])}, {r([repr('_'), repr(' '), repr('-')])})",
           "text.strip()", "text.lower()", "text.upper()",
           f"text.split({r([repr(','), repr(' '), repr(';')])})",
           f"{r([repr(' '), repr('_'), repr('-')])}.join(text)"]
    return f"def clean_{name}(text):\n    return {r(ops)}"


def fam_math():
    name = r(names(1))
    formulas = [
        ("(a, b, c)", "return (a + b + c) / 3"),
        ("(a, b)", "return (a * a + b * b) ** 0.5"),
        ("(base, height)", "return base * height / 2"),
        ("(radius)", "return 3.14159 * radius * radius"),
        ("(principal, rate, years)", "return principal * (1 + rate) ** years"),
        ("(x)", f"return x {r(OPS[:4])} {random.randint(2, 9)} + {random.randint(1, 9)}"),
        ("(values)", "return sum(values) / max(len(values), 1)"),
        ("(a, b)", "return max(a, b) - min(a, b)"),
    ]
    params, ret = r(formulas)
    return f"def calc_{name}{params}:\n    {ret}"


def fam_class():
    name = r(["Point", "Box", "Cell", "Node", "Slot", "Token", "Entry"])
    fields = names(2)
    assign = "\n".join(f"        self.{f} = {f}" for f in fields)
    params = ", ".join(["self"] + fields)
    meth = random.choice(["repr", "len", "add"])
    if meth == "repr":
        m = (f"    def __repr__(self):\n"
             f"        return f\"{name}({fields[0]}={{self.{fields[0]}}})\"")
    elif meth == "len":
        m = "    def __len__(self):\n        return 1"
    else:
        m = (f"    def add(self, other):\n"
             f"        return self.{fields[0]} + other")
    return (f"class {name}:\n    def __init__({params}):\n"
            f"{assign}\n{m}")


def fam_recursive():
    name = r(["depth", "size", "count", "height", "weight"])
    return (f"def tree_{name}(node):\n"
            f"    if node is None:\n        return 0\n"
            f"    children = node.get(\"children\", [])\n"
            f"    return 1 + sum(tree_{name}(c) for c in children)")


def fam_loop_build():
    name = r(names(1))
    start = random.randint(0, 3)
    end = random.randint(5, 15)
    step = r(["", ", 2", ", -1"])
    op = r(["append", "add"])
    return (f"def build_{name}():\n"
            f"    out = []\n"
            f"    for i in range({start}, {end}{step}):\n"
            f"        out.{op}(i {r(OPS[:3])} {random.randint(1, 9)})\n"
            f"    return out" if op == "append"
            else f"def build_{name}():\n"
            f"    out = set()\n"
            f"    for i in range({start}, {end}{step}):\n"
            f"        out.{op}(i {r(OPS[:3])} {random.randint(1, 9)})\n"
            f"    return out")


def fam_exception():
    name = r(names(1))
    return (f"def safe_{name}(x, default=None):\n"
            f"    try:\n"
            f"        return {r(['int(x)', 'float(x)', 'str(x)', 'x[0]', 'len(x)'])}\n"
            f"    except {r(['ValueError', 'TypeError', 'IndexError', 'Exception', 'KeyError'])}:\n"
            f"        return default")


def fam_dict_merge():
    name = r(names(1))
    style = random.randint(0, 2)
    if style == 0:
        return (f"def merge_{name}(a, b):\n"
                f"    out = dict(a)\n    out.update(b)\n    return out")
    if style == 1:
        return (f"def merge_{name}(a, b):\n"
                f"    return {{**a, **b}}")
    return (f"def merge_{name}(a, b):\n"
            f"    out = dict(a)\n"
            f"    for k, v in b.items():\n"
            f"        out[k] = out.get(k, 0) + v\n"
            f"    return out")


def fam_generator():
    name = r(names(1))
    end = random.randint(5, 20)
    expr = r(["i * i", "i * 2", "i + i", "abs(i - 3)"])
    return (f"def gen_{name}(n={end}):\n"
            f"    for i in range(n):\n"
            f"        yield {expr}")


def fam_zip_unpack():
    name = r(names(1))
    return (f"def pair_{name}(keys, vals):\n"
            f"    return {{k: v for k, v in zip(keys, vals)}}")


def fam_nested_get():
    name = r(names(1))
    depth = random.randint(2, 3)
    keys = names(depth)
    path = "".join(f".get({k!r}, {{}})" for k in keys[:-1])
    return (f"def get_{name}(d, default=None):\n"
            f"    cur = d\n"
            f"    for k in {keys!r}:\n"
            f"        if not isinstance(cur, dict):\n"
            f"            return default\n"
            f"        cur = cur.get(k)\n"
            f"    return default if cur is None else cur")


def fam_async():
    name = r(names(1))
    n = r(NOUNS)
    inner = r([
        f"    results = []\n    for x in {n}:\n"
        f"        results.append(await fetch_one(x))\n    return results",
        f"    await asyncio.sleep(0)\n    return {n}",
        f"    async with session.get(url) as resp:\n"
        f"        return await resp.json()",
        f"    tasks = [process(x) for x in {n}]\n"
        f"    return await asyncio.gather(*tasks)"])
    return f"async def {name}_{r(['load','fetch','run','collect'])}({n}):\n{inner}"


def fam_method_class():
    name = r(["Cache", "Queue", "Stack", "Registry", "Buffer", "Store"])
    key = r(names(1))
    return (f"class {name}:\n"
            f"    def __init__(self):\n"
            f"        self._{key} = {{}}\n"
            f"    def get(self, k, default=None):\n"
            f"        return self._{key}.get(k, default)\n"
            f"    def put(self, k, v):\n"
            f"        self._{key}[k] = v\n"
            f"        return v\n"
            f"    def __len__(self):\n"
            f"        return len(self._{key})")


def fam_decorated():
    name = r(names(1))
    dec = r(["cache", "lru_cache(maxsize=128)", "staticmethod",
             "property", "retry(3)", "timer"])
    n = r(NOUNS)
    fn = r(["compute", "load", "parse", "build"])
    expr = r(["sorted(%s)" % n, "len(%s)" % n, "list(%s)" % n,
              "set(%s)" % n])
    return (f"@{dec}\n"
            f"def {name}_{fn}({n}):\n"
            f"    return {expr}")


def fam_with_ctx():
    name = r(names(1))
    if random.random() < 0.5:
        mode = r(['"r"', '"rb"'])
        return (f"def {name}_file(path):\n"
                f"    with open(path, {mode}) as f:\n"
                f"        return f.read()")
    return (f"def {name}_file(path, data):\n"
            f"    with open(path, \"w\") as f:\n"
            f"        f.write(str(data))\n"
            f"    return True")


def fam_while():
    name = r(names(1))
    cond = r(["q", "stack", "todo", "pending"])
    op = r(["pop()", "pop(0)", "popleft()"])
    return (f"def drain_{name}({cond}, visit):\n"
            f"    seen = set()\n"
            f"    while {cond}:\n"
            f"        cur = {cond}.{op}\n"
            f"        if cur in seen:\n"
            f"            continue\n"
            f"        seen.add(cur)\n"
            f"        visit(cur)")


def fam_comprehension():
    name = r(names(1))
    n = r(NOUNS)
    style = random.randint(0, 3)
    if style == 0:
        body = f"return {{x: x * x for x in {n}}}"
    elif style == 1:
        body = f"return {{x for x in {n} if x {r(CMP)} 0}}"
    elif style == 2:
        body = (f"return [y for x in {n}\n"
                f"           for y in x\n"
                f"           if y is not None]")
    else:
        body = f"return [{r(['str(x)', 'int(x)', 'repr(x)'])} for x in {n}]"
    return f"def {name}_map({n}):\n    {body}"


def fam_lambda():
    name = r(names(1))
    n = r(NOUNS)
    return (f"def rank_{name}({n}):\n"
            f"    key = lambda x: (x {r(CMP[:2])} 0, x)\n"
            f"    return sorted({n}, key=key)")


def fam_import():
    name = r(names(1))
    imps = r(["import os", "import sys", "from pathlib import Path",
              "import json", "from collections import defaultdict",
              "import re"])
    return (f"{imps}\n"
            f"def {name}_path(p):\n"
            f"    return str(p).replace(\"\\\\\", \"/\")")


def fam_typed():
    name = r(names(1))
    t1, t2 = random.sample(TYPES, 2)
    default = r(["", " = None", " = 0", ' = ""'])
    none_ret = r(["None", "[]", "{}", "0", '""'])
    body_ret = r(["x", "list(x)", "str(x)", "len(x)"])
    return (f"def typed_{name}(x: {t1}{default}) -> {t2}:\n"
            f"    if x is None:\n"
            f"        return {none_ret}\n"
            f"    return {body_ret}")


def fam_docstring():
    name = r(names(1))
    n = r(NOUNS)
    doc = r(["Return the sorted unique values.",
             "Compute the total, ignoring Nones.",
             "Parse each entry and collect results."])
    body = r([f"return sorted(set({n}))",
              f"return sum(x for x in {n} if x is not None)",
              f"return [parse(x) for x in {n}]"])
    return (f"def {name}_{r(['summarize','collect','parse'])}({n}):\n"
            f"    \"\"\"{doc}\"\"\"\n"
            f"    {body}")


FAMS = [fam_accumulate, fam_transform, fam_filter, fam_validator,
        fam_lookup, fam_guard, fam_string, fam_math, fam_class,
        fam_recursive, fam_loop_build, fam_exception, fam_dict_merge,
        fam_generator, fam_zip_unpack, fam_nested_get,
        fam_async, fam_method_class, fam_decorated, fam_with_ctx,
        fam_while, fam_comprehension, fam_lambda, fam_import,
        fam_typed, fam_docstring]


def main():
    random.seed(SEED)
    seen, out = set(), []
    stats = {"target": N_OUT, "families": len(FAMS), "tries": 0,
             "parse_fail": 0, "dup": 0}
    while len(out) < N_OUT and stats["tries"] < N_OUT * 20:
        stats["tries"] += 1
        src = r(FAMS)()
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
        if src in seen:
            stats["dup"] += 1
            continue
        seen.add(src)
        out.append(src)
    stats["kept"] = len(out)
    with open(OUTD + "/" + OUT_NAME, "w") as fo:
        fo.write("\n\n".join(out))
    json.dump(stats, open(OUTD + "/" +
                          OUT_NAME.replace(".txt", "_stats.json"), "w"),
              indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
