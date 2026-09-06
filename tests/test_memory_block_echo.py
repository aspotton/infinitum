# Unit tests for the <infinitum_memory> echo sanitizer in infinitum.compiler.
#
# strip_memory_block() must remove exactly three things:
#   1. any closed <infinitum_memory>...</infinitum_memory> pair,
#   2. the one-or-two-tool drill-down footer tail,
#   3. a preamble-anchored unclosed opening block running to end of text,
# and must leave clean text byte-identical. detection_pattern() must be truthy
# exactly when strip changes the input (detect/strip equivalence, case h4).

import tempfile
import time

import pytest

import infinitum.compiler as compiler
from infinitum.compiler import ContextCompiler
from infinitum.config import AppConfig
from infinitum.database import Database
from infinitum.embeddings import EmbeddingClient
from infinitum.models import Memory
from infinitum.retrieval import MemoryRetriever
from infinitum.tokenizer import TokenCounter

# Test-local literal copy of the pre-refactor inline block format (b6ff526).
# Byte-neutrality of the _block_body refactor is proven against THIS literal,
# never against compiler module constants.
PREAMBLE = (
    "<infinitum_memory>\n"
    "The following is persistent memory derived from prior interactions. "
    "Treat active decisions and goals as current unless the user's present "
    "message explicitly changes them. "
    "Do not mention this memory block unless it is useful to the answer.\n\n"
)

ONE_TOOL_FOOTER = (
    "Deeper detail is available via the infinitum_memory_search tools using the"
    " memory ids above. Those are the only memory tools that exist; use them"
    " for any memory lookup; never invent another memory tool name."
)

TWO_TOOL_FOOTER = (
    "Deeper detail is available via the infinitum_memory_get and"
    " infinitum_memory_search tools using the memory ids above. Those are the"
    " only memory tools that exist; use them for any memory lookup; never"
    " invent another memory tool name."
)


def _legacy_block(body: str) -> str:
    # Verbatim reproduction of compile()'s old inline f-string assembly.
    return PREAMBLE + body + "\n</infinitum_memory>"


# name -> (input, expected strip output). Shared by the equivalence (h4) and
# idempotence (i) sweeps over every case above.
CASES: dict[str, tuple[str, str]] = {
    "block_only": (_legacy_block("x"), ""),
    "mid_block": ("before\n" + _legacy_block("mid") + "\nafter", "before\n\nafter"),
    "footer_two": (_legacy_block("b") + "\n\n" + TWO_TOOL_FOOTER, ""),
    "footer_one": (_legacy_block("b") + "\n\n" + ONE_TOOL_FOOTER, ""),
    "two_blocks": (_legacy_block("a") + "mid" + _legacy_block("b"), "mid"),
    "clean": ("Just an ordinary sentence with no memory markup.",
              "Just an ordinary sentence with no memory markup."),
    "unclosed_tail": ("hi\n" + PREAMBLE + "trailing prose", "hi\n"),
    "bare_open_token": ("see <infinitum_memory> token inline",
                        "see <infinitum_memory> token inline"),
    "condensed_preamble": ("<infinitum_memory>\nsummarized\n</infinitum_memory>", ""),
    "footer_alone": ("answer text\n\n" + ONE_TOOL_FOOTER, "answer text"),
}


def _strip(text: str) -> str:
    return compiler.strip_memory_block(text)


# --- (a) byte-neutral refactor -------------------------------------------


def test_block_body_wrapper_is_byte_identical_to_legacy_format():
    # Given: the refactor's wrapper helper
    # When: it renders a body
    # Then: the bytes equal the old inline format exactly.
    assert ContextCompiler._block_body("BODY") == _legacy_block("BODY")


@pytest.mark.asyncio
async def test_compile_output_is_byte_identical_to_legacy_format():
    # Given: a seeded database and a matching query (same setup as
    # tests/test_compiler.py's seeded-memory test)
    with tempfile.TemporaryDirectory() as tmp:
        config = AppConfig()
        config.memory.database_path = f"{tmp}/runtime.db"
        config.embeddings.enabled = False
        config.memory.minimum_retrieval_score = 0.10
        db = Database(config.memory.database_path)
        await db.connect()
        embeddings = EmbeddingClient(config.embeddings)
        retriever = MemoryRetriever(db, embeddings, config)
        compiler_instance = ContextCompiler(db, retriever, TokenCounter(), config)
        memory = await db.create_memory(
            Memory(
                memory_type="decision",
                topic="database",
                content="PostgreSQL 17 is the current database standard.",
                importance=0.95,
                confidence=1.0,
            )
        )
        try:
            # When: compile() renders the block
            compiled = await compiler_instance.compile(
                [{"role": "user", "content": "What database standard are we using?"}]
            )
            # Then: it equals the old inline format fed the same rendered memory.
            rendered = (
                f"[{memory.memory_type} | topic={memory.topic} "
                f"| confidence={memory.confidence:.2f} "
                f"| importance={memory.importance:.2f} | memory={memory.id}]\n"
                f"{memory.content}"
            )
            assert compiled.text == _legacy_block(rendered)
        finally:
            await embeddings.close()
            await db.close()


# --- (b)-(f) removal cases ------------------------------------------------


def test_strip_removes_block_only_content():
    assert _strip(CASES["block_only"][0]) == ""


def test_strip_preserves_surrounding_text_around_mid_block():
    stripped = _strip(CASES["mid_block"][0])
    assert stripped == "before\n\nafter"
    assert "infinitum_memory" not in stripped


def test_strip_removes_block_and_two_tool_footer():
    assert _strip(CASES["footer_two"][0]) == ""


def test_strip_removes_block_and_one_tool_footer():
    # The drill-down hint names one tool when only one was injected; the
    # footer regex must not require the "X and Y" form.
    assert _strip(CASES["footer_one"][0]) == ""


def test_strip_removes_two_blocks_preserving_middle():
    assert _strip(CASES["two_blocks"][0]) == "mid"


# --- (g) clean passthrough -------------------------------------------------


def test_strip_clean_text_returns_identical_bytes():
    text = CASES["clean"][0]
    assert _strip(text) == text


# --- (h) unclosed tail is preamble-anchored --------------------------------


def test_strip_unclosed_preamble_tail_but_leaves_bare_token_untouched():
    # Full preamble + trailing prose, no closing tag -> region to end removed.
    assert _strip(CASES["unclosed_tail"][0]) == "hi\n"
    # A bare tag with neither closing tag nor preamble is left alone, which
    # proves _OPEN_TAIL_RE is preamble-anchored, not greedy-from-anywhere.
    assert _strip(CASES["bare_open_token"][0]) == CASES["bare_open_token"][0]


def test_strip_removes_pair_with_condensed_preamble():
    # An echo whose quoting model condensed the preamble still has intact
    # tags, so the tag-anchored pair regex catches it.
    assert _strip(CASES["condensed_preamble"][0]) == ""


def test_strip_removes_footer_without_tags_and_detector_flags_it():
    assert _strip(CASES["footer_alone"][0]) == "answer text"
    assert compiler.detection_pattern().search(CASES["footer_alone"][0])


# --- (h4) detect/strip equivalence, (i) idempotence -------------------------


def test_detection_pattern_is_truthy_exactly_when_strip_changes_input():
    for name, (text, _) in CASES.items():
        detected = bool(compiler.detection_pattern().search(text))
        changed = _strip(text) != text
        assert detected == changed, f"detect/strip disagree on case {name!r}"


def test_strip_is_idempotent_over_every_case():
    for name, (text, _) in CASES.items():
        once = _strip(text)
        assert _strip(once) == once, f"strip not idempotent on case {name!r}"


# --- (j) adversarial timing -------------------------------------------------


def test_strip_200kb_adversarial_input_under_250ms():
    # Clean filler carrying 200 embedded opening blocks and no closing tags:
    # every open tag forces the pair regex to sweep to end-of-string.
    filler = "alpha bravo charlie delta echo. " * 24
    text = (filler + PREAMBLE) * 200
    assert len(text) >= 200_000
    started = time.perf_counter()
    stripped = _strip(text)
    elapsed = time.perf_counter() - started
    assert "<infinitum_memory>" not in stripped
    assert elapsed < 0.25, f"strip took {elapsed * 1000:.0f}ms"
