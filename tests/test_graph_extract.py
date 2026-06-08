import pytest

from ragcore.graph import extract_triples


@pytest.mark.asyncio
async def test_extract_triples_parses_json():
    async def fake_chat(prompt: str) -> str:
        return '{"triples": [{"subject": "Ada Lovelace", "predicate": "wrote", "object": "the first algorithm"}]}'
    triples = await extract_triples("Ada Lovelace wrote the first algorithm.", fake_chat)
    assert triples == [{"subject": "Ada Lovelace", "predicate": "wrote", "object": "the first algorithm"}]


@pytest.mark.asyncio
async def test_extract_triples_tolerates_garbage():
    async def fake_chat(prompt: str) -> str:
        return "I could not find any entities."     # not JSON
    assert await extract_triples("noise", fake_chat) == []


@pytest.mark.asyncio
async def test_extract_triples_strips_code_fences():
    async def fake_chat(prompt: str) -> str:
        return '```json\n{"triples": [{"subject": "A", "predicate": "rel", "object": "B"}]}\n```'
    triples = await extract_triples("x", fake_chat)
    assert triples == [{"subject": "A", "predicate": "rel", "object": "B"}]
