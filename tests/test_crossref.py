from zk import crossref, outline

CLIENT = """namespace Chat.Client
{
    public class ClientController : ControllerBase
    {
        [Requirement("R-06")]
        public void Logout() { }
        public void CloseClient() { }
    }

    public class AdminViewCallbacks
    {
        public Func<string, bool> RenderMonitoring;
        public AdminViewCallbacks() { }
    }
}
"""
REGISTRATION = """namespace Chat.Client
{
    public class RegistrationController
    {
        /// <summary>Checks a new account (R-02).</summary>
        public bool ValidateRegistration(string user) { return true; }
    }
}
"""
SERVICE = """class AuthService(BaseService):
    def login(self, user):
        pass


def answer_query(text):
    pass
"""
BASE = "class BaseService:\n    def close(self):\n        pass\n"


def facts():
    outlines = {
        "Client/ClientController.cs": outline.outline("ClientController.cs", CLIENT),
        "Client/RegistrationController.cs": outline.outline("R.cs", REGISTRATION),
        "api/auth.py": outline.outline("auth.py", SERVICE),
        "api/base.py": outline.outline("base.py", BASE),
    }
    return crossref.facts_from(outlines, [*outlines, "Client/Form1.Designer.cs"])


def test_facts_know_members_owners_files_and_tagged_ids():
    f = facts()
    assert f.members["ClientController"] == {"Logout", "CloseClient"}
    assert f.owners["ValidateRegistration"] == {"RegistrationController"}
    assert "answer_query" in f.file_names["auth.py"]
    assert f.ids == {
        "R-06": ["Client/ClientController.cs: ClientController.Logout"],
        "R-02": ["Client/RegistrationController.cs: RegistrationController.ValidateRegistration"],
    }
    assert "form1.designer.cs" in f.files


def test_mentions_are_ids_and_known_class_names_not_versions():
    text = "R-06 Logout calls ClientController.Logout; see FIX-44, T-54 and UTF-8, SHA-256."
    assert crossref.mentions(text, facts()) == ["R-06", "FIX-44", "T-54", "ClientController"]
    assert crossref.mentions(text) == ["R-06", "FIX-44", "T-54"]


def test_a_member_placed_in_the_wrong_class_or_file_is_stale():
    f = facts()
    assert crossref.stale_reasons("`ClientController.ValidateRegistration()` checks it.", f) == [
        "ValidateRegistration is declared in RegistrationController, not ClientController"
    ]
    assert crossref.stale_reasons("ValidateRegistration in ClientController runs first.", f)
    assert crossref.stale_reasons("answer_query in base.py falls back.", f) == [
        "answer_query is in auth.py, not base.py"
    ]
    assert crossref.stale_reasons("See Client/OldClient.cs for it.", f) == [
        "Client/OldClient.cs is not in the code any more"
    ]
    current = (
        "ClientController.Logout ends the session; ValidateRegistration in "
        "RegistrationController checks it; AuthService.close is inherited; Form1.Designer.cs "
        "is generated; Logout in the UI; Client/ClientController.cs holds it."
    )
    assert crossref.stale_reasons(current, f) == []  # what matches the code is not flagged
    assert crossref.stale_reasons("ClientController.ValidateRegistration", None) == []


def test_what_only_looks_like_a_misplaced_member_is_not_flagged():
    f = facts()
    near_misses = (
        "The view keeps ClientController. Logout then ends the session. "  # two sentences
        "`ClientController`. CloseClient runs next. "
        "AdminViewCallbacks.RenderMonitoring draws it. "  # a delegate field
        "AdminViewCallbacks from the ClientController is built first. "  # a constructor
        "ValidateRegistration of ClientController's form is shown."  # "of" says too little
    )
    assert crossref.stale_reasons(near_misses, f) == []


def test_code_spots_that_name_fewest_ids_come_first():
    tagged = outline.outline(
        "a.cs",
        """public class A
{
    /// <summary>Covers R-01, R-02, R-03 and R-06.</summary>
    public void Many() { }
    [Requirement("R-06")]
    public void One() { }
}
""",
    )
    assert crossref.facts_from({"a.cs": tagged}).ids["R-06"] == ["a.cs: A.One", "a.cs: A.Many"]
