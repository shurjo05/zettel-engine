"""A source file's outline: its classes and functions, their signatures and comments.

What an agent scans a file for before it reads one part of it: which types the file
declares, what each method takes and returns, the comment above it, and any
requirement ids it carries (`[Requirement("R-08")]`, `@Tag("UC-3")`). No bodies, so
an outline changes only when a signature or a comment does.

Python is read with `ast`. Brace languages (C#, Java, Kotlin, C, C++, JavaScript,
TypeScript, Swift, Go, Rust) are read line by line, tracking braces, so only
declarations at type or file level are kept and nothing inside a method body is.
It is a scanner, not a parser: an odd line may be missed, never invented.

Pure: no file access.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

LANGS = {
    ".py": "Python", ".cs": "C#", ".java": "Java", ".kt": "Kotlin", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".c": "C", ".h": "C/C++", ".hpp": "C++",
    ".swift": "Swift", ".go": "Go", ".rs": "Rust",
}  # fmt: skip
# Written by tools, not people: their outline is noise.
GENERATED = (
    "*.designer.cs", "*.g.cs", "*.g.i.cs", "assemblyinfo.cs", "*.assemblyattributes.cs",
    "*.min.js", "*.bundle.js", "*_pb2.py", "*_pb2_grpc.py", "*.pb.go", "*.generated.*",
    "*modelsnapshot.cs", "migrations/*", "*/migrations/*",
)  # fmt: skip
DOC_CHARS = 200  # a comment is cut to its first sentence, at most this long
ATTR_CHARS = 120
MAX_VALUES = 30  # enum members listed

# Requirement, use-case and test ids: R-08, UC-3, FR-12, NFR-2a.
IDS = re.compile(r"\b[A-Z]{1,4}-\d{1,4}[a-z]?\b")


@dataclass
class Item:
    kind: str  # "type" or "member"
    signature: str
    doc: str = ""
    attrs: list[str] = field(default_factory=list)
    depth: int = 0  # how many types it is inside
    values: list[str] = field(default_factory=list)  # an enum's members

    @property
    def ids(self) -> list[str]:
        return list(dict.fromkeys(IDS.findall(" ".join([*self.attrs, self.doc]))))


@dataclass
class Outline:
    language: str
    lines: int
    summary: str = ""  # the file's own opening comment or docstring
    items: list[Item] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        found = IDS.findall(self.summary)
        for item in self.items:
            found += item.ids
        return list(dict.fromkeys(found))


def language(name: str) -> str | None:
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return LANGS.get(suffix)


def outline(name: str, text: str) -> Outline | None:
    """The outline of a source file, or None when it isn't code or declares nothing."""
    lang = language(name)
    if lang is None:
        return None
    result = _python(text) if lang == "Python" else _braces(text, lang)
    if result is None or not result.items:
        return None
    result.language = lang
    result.lines = text.count("\n") + 1
    return result


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", text)
    text = match[1] if match else text
    return text if len(text) <= DOC_CHARS else text[: DOC_CHARS - 3].rsplit(" ", 1)[0] + "..."


# Python.


def _python(text: str) -> Outline | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    out = Outline("Python", 0, _first_sentence(ast.get_docstring(tree) or ""))

    def visit(body: list[ast.stmt], depth: int) -> None:
        for node in body:
            attrs = ["@" + ast.unparse(d)[:ATTR_CHARS] for d in getattr(node, "decorator_list", [])]
            doc = ""
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                doc = _first_sentence(ast.get_docstring(node) or "")
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases + node.keywords)
                sig = f"class {node.name}" + (f"({bases})" if bases else "")
                out.items.append(Item("type", sig, doc, attrs, depth))
                visit(node.body, depth + 1)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name.startswith("__") and node.name != "__init__":
                    continue
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                sig = f"{prefix} {node.name}({ast.unparse(node.args)})"
                if node.returns is not None:
                    sig += f" -> {ast.unparse(node.returns)}"
                out.items.append(Item("member", sig, doc, attrs, depth))

    visit(tree.body, 0)
    return out


# Brace languages.

_MODIFIERS = (
    r"(?:(?:public|private|protected|internal|static|abstract|sealed|partial|final|export|"
    r"default|declare|readonly|data|open|inner|pub(?:\([\w:]+\))?|unsafe|async|new|override|"
    r"virtual|extern|inline|file|ref|fileprivate|required|annotation|companion)\s+)*"
)
_TYPE = re.compile(
    rf"^\s*{_MODIFIERS}(class|interface|struct|enum(?:\s+class|\s+struct)?|"
    r"record(?:\s+class|\s+struct)?|trait|protocol|extension|impl|@interface|object|"
    r"namespace|mod|module)\b\s*(.*)$"
)
_GO_TYPE = re.compile(r"^\s*type\s+(\w+)\s+(struct|interface)\b")
_GO_FUNC = re.compile(r"^\s*func\s*(\([^)]*\)\s*)?(\w+)\s*[\[(]")
_ARROW = re.compile(
    r"^\s*(?:export\s+)?(?:(?:const|let|var)\s+)?(?:(?:public|private|protected|static|"
    r"readonly)\s+)*([A-Za-z_$][\w$]*)\s*(?::\s*[^=]+)?=\s*(?:async\s+)?"
    r"(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::\s*[^=]+)?=>"
)
_PROPERTY = re.compile(r"^\s*[\w<>\[\],.?\s]+?\s([A-Za-z_]\w*)\s*\{\s*(?:get|set|init)\b")
_EXPR_PROPERTY = re.compile(r"^\s*[\w<>\[\],.?\s]+?\s([A-Za-z_]\w*)\s*=>")
_NOT_DECL = {
    "if", "for", "foreach", "while", "switch", "catch", "using", "return", "throw", "new",
    "await", "lock", "else", "do", "try", "case", "typeof", "sizeof", "yield", "base", "this",
    "super", "fixed", "checked", "unchecked", "when", "match", "loop", "assert", "delete",
    "goto", "nameof", "default", "static_assert", "defer", "go", "select", "print", "echo",
}  # fmt: skip


def _mask(lines: list[str]) -> list[str]:
    """Each line with strings' insides and comments blanked, same length, so braces and
    parentheses in them don't count and positions still match the original."""
    out: list[str] = []
    in_block = False
    for line in lines:
        chars = list(line)
        i, n = 0, len(chars)
        while i < n:
            if in_block:
                if line.startswith("*/", i):
                    chars[i] = chars[i + 1] = " "
                    in_block = False
                    i += 2
                    continue
                chars[i] = " "
                i += 1
            elif line.startswith("/*", i):
                in_block = True
                chars[i] = chars[i + 1] = " "
                i += 2
            elif line.startswith("//", i):
                for j in range(i, n):
                    chars[j] = " "
                break
            elif chars[i] in "\"'`":
                quote, j = chars[i], i + 1
                while j < n and line[j] != quote:
                    if line[j] == "\\":
                        chars[j] = "_"
                        j += 1
                        if j < n:
                            chars[j] = "_"
                        j += 1
                        continue
                    chars[j] = "_"
                    j += 1
                i = j + 1
            else:
                i += 1
        out.append("".join(chars))
    return out


def _doc_text(comment: list[str]) -> str:
    text = " ".join(
        re.sub(r"^\s*(?:///?|/\*\*?|\*/|\*)\s?", "", c).removesuffix("*/") for c in comment
    )
    text = re.sub(r"<see\s+cref=\"([^\"]+)\"\s*/>", r"\1", text)
    text = re.sub(r"<param\b.*", "", text, flags=re.S)  # keep the summary, not the params
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(?:^|\s)@(?:param|return|returns|throws|exception)\b.*", "", text, flags=re.S)
    return _first_sentence(text)


def _cut(raw: str, masked: str) -> str:
    """A declaration without its body: up to the `{`, `=>` or `;` outside parentheses."""
    depth = 0
    for i, ch in enumerate(masked):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth <= 0 and (ch in "{;" or masked.startswith("=>", i)):
            return " ".join(raw[:i].split())
    return " ".join(raw.split())


_ATTR_PREFIX = re.compile(
    r"^\s*(?:\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]\s*|#\[[^\]]*\]\s*|@[\w.]+(?:\([^()]*\))?(?:\s+|$))+"
)
MAX_JOIN = 12  # lines a declaration may run over


def _braces(text: str, lang: str) -> Outline:
    raw_lines = text.splitlines()
    masked_lines = _mask(raw_lines)
    out = Outline(lang, 0)
    stack: list[str] = []  # "ns", "type", "enum" or "body" for each open brace
    pending: str | None = None  # what the next "{" opens
    comment: list[str] = []
    attrs: list[str] = []
    seen_code = False
    i = 0
    while i < len(raw_lines):
        raw, masked = raw_lines[i], masked_lines[i]
        i += 1
        stripped = raw.strip()
        if not all(k in ("ns", "type", "enum") for k in stack):  # inside a body
            pending = _count(masked, stack, pending)
            continue
        if not masked.strip():
            if stripped:  # a comment line
                comment.append(stripped)
            else:  # a blank line ends a comment that isn't above anything
                if not seen_code and comment and not out.summary:
                    out.summary = _file_summary(comment)
                if not attrs:
                    comment = []
            continue
        if stripped.startswith("#") and not stripped.startswith("#["):
            continue  # preprocessor, shebang
        # A declaration may run over several lines: join until its parentheses close.
        joined = 0
        while masked.count("(") > masked.count(")") and i < len(raw_lines) and joined < MAX_JOIN:
            raw += " " + raw_lines[i].strip()
            masked += " " + masked_lines[i].strip()
            i += 1
            joined += 1
        seen_code = True
        if prefix := _ATTR_PREFIX.match(masked):  # `[HttpPost]`, `@Override`, `#[test]`
            attrs.append(" ".join(raw[: prefix.end()].split())[:ATTR_CHARS])
            rest_raw, rest_masked = raw[prefix.end() :], masked[prefix.end() :]
            if not rest_masked.strip():
                pending = _count(masked, stack, pending)
                continue
        else:
            rest_raw, rest_masked = raw, masked
        scope = stack[-1] if stack else "ns"
        item, opens = _declaration(rest_raw, rest_masked, lang, scope)
        if item is not None:
            item.doc = _doc_text(comment) if comment else ""
            item.attrs = attrs
            item.depth = stack.count("type") + stack.count("enum")
            if opens != "ns":  # a namespace scopes what follows; it isn't listed itself
                out.items.append(item)
            if opens == "enum" and "{" in rest_masked:  # `enum Status { On, Off }`
                item.values = _enum_values(rest_masked[rest_masked.find("{") + 1 :])
            pending = opens
        elif scope == "enum":
            enum = next((x for x in reversed(out.items) if x.kind == "type"), None)
            if enum is not None:
                enum.values += _enum_values(rest_masked)[: MAX_VALUES - len(enum.values)]
        comment, attrs = [], []
        pending = _count(masked, stack, pending)
    return out


def _enum_values(masked: str) -> list[str]:
    """The member names in an enum's text, up to its first `;` or `}`."""
    text = re.split(r"[;}]", masked, maxsplit=1)[0]
    return re.findall(r"(?:^|,)\s*([A-Za-z_]\w*)", text)[:MAX_VALUES]


def _file_summary(comment: list[str]) -> str:
    text = _doc_text(comment)
    return "" if re.search(r"copyright|licen[cs]e|spdx", text, re.I) else text


def _count(masked: str, stack: list[str], pending: str | None) -> str | None:
    """Push and pop scopes for the braces on a line; the first "{" opens `pending`."""
    for ch in masked:
        if ch == "{":
            stack.append(pending or "body")
            pending = None
        elif ch == "}" and stack:
            stack.pop()
        elif ch == ";" and pending == "body":
            pending = None  # `void Run();` declared, no body follows
    return pending


def _declaration(raw: str, masked: str, lang: str, scope: str) -> tuple[Item | None, str | None]:
    """The item a line declares and what its "{" opens, or (None, None)."""
    sig = _cut(raw, masked)
    ends = masked.rstrip()
    if lang == "Go":
        if _GO_TYPE.match(masked):
            return Item("type", sig), "type"
        if _GO_FUNC.match(masked):
            return Item("member", sig), "body"
        return None, None
    if m := _TYPE.match(masked):
        word = m[1].split()[0]
        if word in ("namespace", "mod", "module"):
            return Item("type", sig), None if ends.endswith(";") else "ns"
        if ends.endswith(";") and "{" not in masked and word != "record":
            return None, None  # a forward declaration: `class Foo;`
        return Item("type", sig), "enum" if word == "enum" else "type"
    in_type = scope == "type"
    script = lang in ("JavaScript", "TypeScript")
    if script and _ARROW.match(masked):
        return Item("member", sig), "body"
    if lang == "C#" and in_type:
        if _PROPERTY.match(masked):
            return Item("member", sig + " { " + _accessors(masked) + " }"), "body"
        if _EXPR_PROPERTY.match(masked) and "(" not in masked.split("=>", 1)[0]:
            return Item("member", sig), "body"
    paren = masked.find("(")
    before = masked[:paren].rstrip() if paren > 0 else ""
    words = before.split()
    if not words or "=" in before or "." in words[-1]:
        return None, None  # no name, an assignment, or a call on something (`app.use(`)
    name = re.search(r"(~?[A-Za-z_$][\w$]*|operator\s*\S+)\s*(?:<[^()]*>)?$", before)
    if not name or name[1] in _NOT_DECL or words[0] in _NOT_DECL:
        return None, None
    words = before[: name.start()].split()
    tail = masked[masked.rfind(")") + 1 :].strip()
    if not in_type:
        # At file level a call (`describe("x", () => {`, `TEST(A, B) {`) looks like a
        # declaration: a declaration has a return type or keyword before its name.
        if not words:
            return None, None
        if not (
            ends.endswith(("{", ")", ";"))
            or tail.startswith(
                ("=>", "const", "->", ":", "noexcept", "override", "where", "throws")
            )
        ):
            return None, None
    elif script and not words and not (ends.endswith((")", ";")) or tail.startswith(("{", ":"))):
        return None, None
    return Item("member", sig), None if ends.endswith(";") else "body"


def _accessors(masked: str) -> str:
    """`get; set;`, `get; private set;` from a property line."""
    inside = masked[masked.find("{") + 1 :]
    found = re.findall(r"((?:(?:private|protected|internal|public)\s+)?(?:get|set|init))\b", inside)
    return " ".join(f"{a};" for a in found)


# As Markdown.

_TYPE_NAME = re.compile(
    r"\b(class|interface|struct|enum|record|trait|protocol|extension|impl|object|type)"
    r"\s+(?:class\s+|struct\s+)?([A-Za-z_]\w*)"
)


def markdown(o: Outline) -> str:
    """The outline as Markdown: a `##` heading per type, a bullet per member."""
    lines: list[str] = []
    scopes: list[tuple[str, str]] = []  # (kind, name) of the enclosing types, by depth
    heading_depth = 0  # members right under the last heading are at this depth
    for item in o.items:
        code = " ".join([*item.attrs, item.signature]).replace("`", "'")
        if item.kind == "type":
            m = _TYPE_NAME.search(item.signature)
            kind, name = (m[1], m[2]) if m else ("type", item.signature.split("(")[0].split()[-1])
            scopes = [*scopes[: item.depth], (kind, name)]
            lines += ["", f"## {kind} {'.'.join(n for _, n in scopes)}", "", f"`{code}`"]
            if item.doc:
                lines.append(item.doc)
            if item.values:
                lines.append("Values: " + ", ".join(item.values) + ".")
            lines.append("")
            heading_depth = item.depth + 1
            continue
        if item.depth != heading_depth:  # back out of a nested type, or to file level
            if item.depth == 0:
                lines += ["", "## Functions", ""]
            else:
                outer = scopes[: item.depth]
                lines += ["", f"## {outer[-1][0]} {'.'.join(n for _, n in outer)} (continued)", ""]
            heading_depth = item.depth
        lines.append(f"- `{code}`" + (f": {item.doc}" if item.doc else ""))
    return "\n".join(lines).strip() + "\n"
