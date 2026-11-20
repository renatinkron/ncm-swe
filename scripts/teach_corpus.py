#!/usr/bin/env python3
"""Devin's own curriculum for the NCM — complete micro-lessons, not random
file slices. Each lesson is a self-contained unit sized for the 128-token
segment window (~350 chars): the model sees WHOLE patterns, never fragments.

Covers: signature/return, arithmetic, strings, lists, dicts, loops,
conditionals, comprehensions, classes, try/except, simple algorithms,
and intent->code pairs (docstring states goal, body solves it).
"""
import os
import random

random.seed(7)
OUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "corpus_teach")

lessons = []

# --- family 1: arithmetic functions (intent in docstring + body) ---
NAMES = ["add", "sum_two", "total", "combine", "plus"]
OPS = [("+", "adds"), ("-", "subtracts"), ("*", "multiplies")]
for n in NAMES:
    for op, verb in OPS:
        lessons.append(
            f'def {n}(a, b):\n    """{verb.capitalize()} two numbers."""\n'
            f'    return a {op} b\n')
        lessons.append(
            f'def {n}_three(a, b, c):\n    """{verb.capitalize()} three '
            f'numbers."""\n    return a {op} b {op} c\n')

# --- family 2: string ops ---
for name in ["greet", "welcome", "hello_user"]:
    lessons.append(
        f'def {name}(name):\n    """Return a greeting for name."""\n'
        f"    return 'Hello, ' + name + '!'\n")
lessons += [
    'def shout(text):\n    """Uppercase the text."""\n    return text.upper()\n',
    "def whisper(text):\n    return text.lower()\n",
    'def initials(name):\n    """First letter of each word."""\n'
    '    return "".join(w[0] for w in name.split())\n',
    "def join_words(words, sep=' '):\n    return sep.join(words)\n",
    'def repeat(text, n):\n    """Repeat text n times."""\n    return text * n\n',
]

# --- family 3: lists ---
lessons += [
    "def first(items):\n    return items[0]\n",
    "def last(items):\n    return items[-1]\n",
    'def count(items, target):\n    """How many times target appears."""\n'
    '    return items.count(target)\n',
    'def total(numbers):\n    """Sum of a list."""\n    s = 0\n'
    '    for n in numbers:\n        s += n\n    return s\n',
    'def maximum(numbers):\n    m = numbers[0]\n    for n in numbers:\n'
    '        if n > m:\n            m = n\n    return m\n',
    'def squares(n):\n    """First n squares."""\n'
    '    return [i * i for i in range(n)]\n',
    'def evens(numbers):\n    return [n for n in numbers if n % 2 == 0]\n',
    'def double_all(items):\n    return [x * 2 for x in items]\n',
]

# --- family 4: conditionals ---
lessons += [
    'def sign(x):\n    """Sign of x: -1, 0 or 1."""\n    if x > 0:\n'
    '        return 1\n    elif x < 0:\n        return -1\n    return 0\n',
    'def clamp(x, lo, hi):\n    """Keep x inside [lo, hi]."""\n'
    '    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n'
    '    return x\n',
    'def is_adult(age):\n    return age >= 18\n',
    'def grade(score):\n    if score >= 90:\n        return "A"\n'
    '    if score >= 60:\n        return "B"\n    return "F"\n',
    'def abs_val(x):\n    if x < 0:\n        return -x\n    return x\n',
]

# --- family 5: dicts ---
lessons += [
    "def get_name(user):\n    return user.get('name', 'anon')\n",
    'def merge(a, b):\n    """Merge two dicts."""\n    out = dict(a)\n'
    '    out.update(b)\n    return out\n',
    'def invert(d):\n    return {v: k for k, v in d.items()}\n',
    'def word_count(text):\n    """Count each word."""\n    c = {}\n'
    '    for w in text.split():\n        c[w] = c.get(w, 0) + 1\n'
    '    return c\n',
]

# --- family 6: loops + while ---
lessons += [
    'def countdown(n):\n    """n down to 1."""\n    while n > 0:\n'
    '        print(n)\n        n -= 1\n',
    'def factorial(n):\n    """n!"""\n    r = 1\n    for i in range(2, n + 1):\n'
    '        r *= i\n    return r\n',
    'def fib(n):\n    """First n fibonacci numbers."""\n    a, b = 0, 1\n'
    '    out = []\n    for _ in range(n):\n        out.append(a)\n'
    '        a, b = b, a + b\n    return out\n',
]

# --- family 7: classes ---
lessons += [
    'class Point:\n    def __init__(self, x, y):\n        self.x = x\n'
    '        self.y = y\n',
    'class Stack:\n    def __init__(self):\n        self.items = []\n'
    '    def push(self, x):\n        self.items.append(x)\n'
    '    def pop(self):\n        return self.items.pop()\n',
    'class Counter:\n    def __init__(self):\n        self.n = 0\n'
    '    def tick(self):\n        self.n += 1\n        return self.n\n',
]

# --- family 8: try/except ---
lessons += [
    'def to_int(s):\n    """Parse int or return 0."""\n    try:\n'
    '        return int(s)\n    except ValueError:\n        return 0\n',
    'def safe_div(a, b):\n    try:\n        return a / b\n'
    '    except ZeroDivisionError:\n        return None\n',
]

# --- family 9: intent -> code pairs (agentic pattern seed) ---
lessons += [
    '# Task: check if a number is even\ndef is_even(n):\n    return n % 2 == 0\n',
    '# Task: reverse a string\ndef reverse(s):\n    return s[::-1]\n',
    '# Task: longest of two strings\ndef longer(a, b):\n'
    '    return a if len(a) >= len(b) else b\n',
    '# Task: average of a list\ndef average(nums):\n'
    '    return sum(nums) / len(nums)\n',
]

# --- variations: rename + retype to fight memorization ---
varied = []
for les in lessons:
    varied.append(les)
    if les.startswith("def "):
        v = les.replace("(a, b)", "(x, y)").replace(" a ", " x ").replace(
            " b ", " y ")
        varied.append(v)
lessons = varied

os.makedirs(OUT, exist_ok=True)
random.shuffle(lessons)
text = "\n\n".join(lessons)
with open(os.path.join(OUT, "lessons.txt"), "w") as f:
    f.write(text)
print(f"{len(lessons)} lessons, {len(text)} chars -> {OUT}/lessons.txt")
