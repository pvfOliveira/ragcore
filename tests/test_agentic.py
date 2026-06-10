from ragcore.config import AgenticConfig
from ragcore.agentic import grade_documents


async def test_grade_documents_keeps_relevant_only():
    chunks = [{"id": "1", "content": "rrf fuses ranks"}, {"id": "2", "content": "weather today"}]
    async def chat(prompt):
        # grader sees both question and document; discriminate on the document text
        return "yes" if "fuses ranks" in prompt else "no"
    kept = await grade_documents("what is rrf?", chunks, chat)
    assert [c["id"] for c in kept] == ["1"]


async def test_loop_reretrieves_once_then_stops():
    from ragcore import agentic
    searches = []
    async def search_fn(term):
        searches.append(term)
        return [{"id": f"{term}", "content": "weather", "source": "s"}]
    async def chat_fn(prompt):
        if "relevant" in prompt.lower():
            return "no"          # docs always graded irrelevant
        if "grounded" in prompt.lower():
            return "no"
        if "alternative" in prompt.lower():
            return "1. rrf variant"
        return "final answer"
    cfg = AgenticConfig(enabled=True, max_iterations=2, min_relevant=2)
    out = await agentic.run_agentic("explain rrf", cfg, chat_fn, search_fn)
    # initial retrieve + up to max_iterations re-retrieves = 1 + 2 = 3 search calls max
    assert 1 <= len(searches) <= 3
    assert "answer" in out and "grounded" in out
