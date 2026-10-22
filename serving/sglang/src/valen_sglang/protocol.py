"""System One's standard fields with Valen's native score-level budget."""

from typing import Annotated, Union

from pydantic import Field

from sglang.srt.entrypoints.openai.protocol import RequiredDecisionText
from sglang.srt.entrypoints.systemone.protocol import (
    SystemOneChoiceQuestion,
    SystemOneNoulQuestion,
)
from sglang.srt.entrypoints.systemone.protocol import (
    SystemOneRequest as StockSystemOneRequest,
)
from sglang.srt.entrypoints.systemone.protocol import (
    SystemOneScoreQuestion as StockSystemOneScoreQuestion,
)


class SystemOneScoreQuestion(StockSystemOneScoreQuestion):
    criteria: list[RequiredDecisionText] = Field(min_length=1, max_length=255)


Question = Annotated[
    Union[SystemOneNoulQuestion, SystemOneChoiceQuestion, SystemOneScoreQuestion],
    Field(discriminator="type"),
]


class SystemOneRequest(StockSystemOneRequest):
    questions: dict[str, Question] = Field(min_length=1)
