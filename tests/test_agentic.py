from ragcore.config import AgenticConfig
from ragcore.agentic import grade_documents


async def test_grade_documents_keeps_relevant_only():
    chunks = [{"id": "1", "content": "rrf fuses ranks"}, {"id": "2", "content": "weather today"}]
    async def chat(prompt):
        # grader sees both question and document; discriminate on the document text
        return "yes" if "fuses ranks" in prompt else "no"
    kept = await grade_documents("what is rrf?", chunks, chat)
    assert [c["id"] for c in kept] == ["1"]
