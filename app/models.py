from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., max_length=1000, examples=["What does the policy say about topic X?"])

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class Status(str, Enum):
    answered = "answered"
    refused = "refused"


class RefusalReason(str, Enum):
    no_documents = "no_documents"
    low_relevance = "low_relevance"
    not_in_policy = "not_in_policy"
    ungrounded = "ungrounded"
    llm_error = "llm_error"


class Citation(BaseModel):
    document: str = Field(..., description="Document title, e.g. 'Example Policy'")
    section: str = Field(..., description="Heading path, e.g. '2. Section > 2.1 Subsection'")
    quote: str = Field(..., description="Verbatim supporting text, verified server-side")
    doc_id: str
    doc_version: str
    chunk_id: str


class RetrievalDebug(BaseModel):
    top_vector_score: float
    chunks: list[str]


class AskResponse(BaseModel):
    status: Status
    answer: str
    citations: list[Citation] = []
    refusal_reason: RefusalReason | None = None
    retrieval: RetrievalDebug | None = None


class UploadResponse(BaseModel):
    doc_id: str
    title: str
    version: str
    chunks: int
    status: str


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    filename: str
    version: str
    uploaded_at: str
    chunks: int


class LLMCitation(BaseModel):
    source: str
    quote: str


class LLMAnswer(BaseModel):
    status: Status
    answer: str = ""
    citations: list[LLMCitation] = []
