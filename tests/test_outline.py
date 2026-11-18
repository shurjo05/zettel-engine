from zk import outline

CSHARP = """// Chat501 contact handling.

using System;

namespace Chat501.Server
{
    /// <summary>
    /// Handles contact requests from clients.
    /// </summary>
    [Requirement("R-08")]
    public sealed class ContactController : ControllerBase
    {
        private readonly Dictionary<string, Contact> _contacts = new();
        public string Owner { get; private set; }

        /// <summary>Adds a contact. Returns false if present.</summary>
        /// <param name="name">The user name.</param>
        [Requirement("R-08")]
        [HttpPost("add")]
        public bool Add(string name,
                        bool notify = true)
        {
            if (_contacts.ContainsKey(name)) { return false; }
            foreach (var x in _contacts) { Console.WriteLine("}"); }
            void Local(int y) { }
            return true;
        }

        [Requirement("R-09")] public void Remove(string name) => _contacts.Remove(name);

        public enum Status { Online, Away = 2, Offline }

        private class Inner
        {
            internal void Ping() { }
        }

        public void Dispose() { }
    }

    public interface IChatConnection
    {
        /// Logs in with a user name and password hash.
        bool Login(string user, string hash);
    }
}
"""


def test_csharp_outline_keeps_declarations_comments_and_requirement_ids():
    o = outline.outline("Server/ContactController.cs", CSHARP)
    assert o.language == "C#" and o.summary == "Chat501 contact handling."
    sigs = [i.signature for i in o.items]
    assert sigs == [
        "public sealed class ContactController : ControllerBase",
        "public string Owner { get; private set; }",
        "public bool Add(string name, bool notify = true)",
        "public void Remove(string name)",
        "public enum Status",
        "private class Inner",
        "internal void Ping()",
        "public void Dispose()",
        "public interface IChatConnection",
        "bool Login(string user, string hash)",
    ]  # nothing from inside a method body, the brace in a string included
    add = o.items[2]
    assert add.doc == "Adds a contact." and add.attrs == [
        '[Requirement("R-08")]',
        '[HttpPost("add")]',
    ]
    assert o.items[4].values == ["Online", "Away", "Offline"]
    assert o.ids == ["R-08", "R-09"]
    md = outline.markdown(o)
    assert "## class ContactController.Inner" in md
    assert "## class ContactController (continued)\n\n- `public void Dispose()`" in md
    assert '- `[Requirement("R-09")] public void Remove(string name)`' in md


def test_script_and_python_outlines_skip_calls():
    ts = """import express from "express";
/** Builds the router. */
export function makeRouter(db: Db): Router {
  const r = express.Router();
  return r;
}
export const handler = async (event: Event): Promise<void> => {
  await go();
};
describe("x", () => {
  it("works", () => {});
});
app.listen(3000);
export class AppComponent {
  constructor(private http: HttpClient) {}
  ngOnInit(): void {
    this.load();
  }
}
"""
    o = outline.outline("src/app.ts", ts)
    assert [i.signature for i in o.items] == [
        "export function makeRouter(db: Db): Router",
        "export const handler = async (event: Event): Promise<void>",
        "export class AppComponent",
        "constructor(private http: HttpClient)",
        "ngOnInit(): void",
    ]
    assert o.items[0].doc == "Builds the router."
    py = """\"\"\"Sends messages.\"\"\"

class Client(Base):
    \"\"\"A client.\"\"\"

    def __init__(self, host: str):
        pass

    def __repr__(self):
        return ""

    @property
    def name(self) -> str:
        return ""

async def main():
    pass
"""
    o = outline.outline("client.py", py)
    assert o.summary == "Sends messages."
    assert [(i.signature, i.depth) for i in o.items] == [
        ("class Client(Base)", 0),
        ("def __init__(self, host: str)", 1),
        ("def name(self) -> str", 1),
        ("async def main()", 0),
    ]
    assert outline.outline("notes.txt", "class A {}") is None
    assert outline.outline("empty.cs", "// nothing\n") is None
