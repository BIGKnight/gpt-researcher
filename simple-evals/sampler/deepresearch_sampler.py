from typing import Any

from gpt_researcher import GPTResearcher

from ..types import MessageList, SamplerBase

class DeepResearchSampler(SamplerBase):
    """
    Sample from OpenAI's responses API
    """

    def __init__(
        self,
    ):
        pass

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    async def __call__(self, question: MessageList) -> str:
        researcher = GPTResearcher(query=question, report_type="deep")
        research_result = await researcher.conduct_research()
        # Write the report
        answer = await researcher.write_answer()
        return answer
