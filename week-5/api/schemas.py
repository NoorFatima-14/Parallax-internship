"""Request and response models. They also produce the OpenAPI documentation at /docs."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_QUESTION_CHARS = 4000  # longer text is rejected (422). Between 1,000 and 4,000 it is cut to 1,000 (see warnings).

Category = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=40,
                                            pattern=r"^[A-Za-z][A-Za-z.\-]*$")]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


# ------------------------------------------------------------------ requests
class FilterParams(BaseModel):
    """Metadata filters from Week 4. Different kinds are combined with AND, several values of one kind with OR."""

    model_config = ConfigDict(extra="forbid")

    topics: list[Name | int] = Field(default_factory=list, max_length=10,
                                     description="Topic ids or words from a topic label (see GET /topics)")
    categories: list[Category] = Field(default_factory=list, max_length=10,
                                       description="arXiv categories, for example cs.CV")
    datasets: list[Name] = Field(default_factory=list, max_length=10,
                                 description="Dataset names, for example COCO (see GET /datasets)")
    datasets_mode: Literal["any", "all"] = "any"
    year_from: int | None = Field(None, ge=1990, le=2100)
    year_to: int | None = Field(None, ge=1990, le=2100)
    core_only: bool = Field(False, description="Only papers inside a found topic cluster")
    exclude_short: bool = Field(False, description="Leave out the shortest abstracts")


class AskRequest(BaseModel):
    """Body of POST /ask: a question, and optional top_k and filters."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS,
                          examples=["How do diffusion models generate images?"])
    top_k: int | None = Field(None, ge=1, le=20, description="Passages to retrieve (default: the server setting)")
    filters: FilterParams | None = None


class SearchRequest(BaseModel):
    """Body of POST /search: a query, and optional top_k and filters."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS, examples=["uncertainty estimation"])
    top_k: int | None = Field(None, ge=1, le=20)
    filters: FilterParams | None = None


# ------------------------------------------------------------------ responses
class Source(BaseModel):
    """One passage the answer was built from."""

    n: int
    id: str | None = None
    doc_id: str
    title: str = ""
    category: str = ""
    score: float | None = None


class GroundingSummary(BaseModel):
    """Result of the hallucination checks. It holds counts and codes only, never text from the
    model's answer, so an answer that was blocked cannot leak through this summary."""

    level: str
    grounded_fraction: float
    sentences_checked: int
    citations: list[int]
    invalid_citations: list[int]
    unsupported_number_count: int
    block_reasons: list[str]


class AskResponse(BaseModel):
    """Answer of POST /ask. `status` says whether it is an answer, a refusal or a blocked answer."""

    request_id: str
    status: Literal["answered", "answered_with_warning", "refused_by_model",
                    "refused_out_of_domain", "blocked_ungrounded"]
    answer: str
    sources: list[Source]
    warnings: list[str]
    grounding: GroundingSummary | None = None
    model: str | None = None
    top_score: float | None = None
    timings_ms: dict[str, float]
    filter_applied: dict[str, Any] | None = None


class SearchHit(BaseModel):
    """One passage found by a search, with its score and paper details."""

    id: str
    doc_id: str
    title: str = ""
    category: str = ""
    score: float
    text: str
    topic_id: int | None = None
    topic_label: str | None = None
    published_year: int | None = None
    datasets: str | None = None


class SearchResponse(BaseModel):
    """Answer of POST /search."""

    request_id: str
    query: str
    count: int
    hits: list[SearchHit]
    warnings: list[str]
    timings_ms: dict[str, float]
    filter_applied: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Answer of GET /health."""

    status: Literal["ok"]
    version: str
    uptime_seconds: float


class ReadyResponse(BaseModel):
    """Answer of GET /ready."""

    request_id: str
    ready: bool
    chunks: int
    embedding_model: str | None = None
    llm_provider: str | None = None
    llm_models: list[str]
    topics_loaded: bool
    datasets_loaded: bool
    settings: dict[str, Any]


class TopicInfo(BaseModel):
    """One topic that can be used in a filter."""

    topic_id: int
    label: str
    size: int | None = None


class TopicsResponse(BaseModel):
    """Answer of GET /topics."""

    available: bool
    topics: list[TopicInfo]


class DatasetInfo(BaseModel):
    """One dataset name that can be used in a filter, with the number of papers that mention it."""

    name: str
    docs: int | None = None


class DatasetsResponse(BaseModel):
    """Answer of GET /datasets."""

    available: bool
    datasets: list[DatasetInfo]


class ErrorBody(BaseModel):
    """The details of an error: a short code, a message, and the request id."""

    code: str
    message: str
    request_id: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    """The body of every error response."""

    error: ErrorBody
