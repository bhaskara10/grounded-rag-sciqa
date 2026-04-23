"""
State machine transition validators for the ingestion pipeline.

Valid transitions
-----------------
Parse / Enrichment (terminal at COMPLETE):
  pending  -> running
  running  -> complete | failed
  failed   -> running              (retry)
  complete -> (nothing)            terminal

Projection (NOT terminal at COMPLETE — re-projection after enrichment):
  pending  -> running
  running  -> complete | failed
  failed   -> running              (retry)
  complete -> running              (re-project after enrichment)

Usage
-----
Always call validate_*_transition() before writing a new status value.
Never compare statuses inline — centralising the logic here lets you
add guards, logging, or metrics in one place.
"""
from .enums import EnrichmentStatus, ParseStatus, ProjectionStatus

# ── transition tables ─────────────────────────────────────────────────────────

VALID_PARSE_TRANSITIONS: dict[ParseStatus, set[ParseStatus]] = {
    ParseStatus.PENDING:  {ParseStatus.RUNNING},
    ParseStatus.RUNNING:  {ParseStatus.COMPLETE, ParseStatus.FAILED},
    ParseStatus.FAILED:   {ParseStatus.RUNNING},
    ParseStatus.COMPLETE: set(),                        # terminal
}

VALID_ENRICHMENT_TRANSITIONS: dict[EnrichmentStatus, set[EnrichmentStatus]] = {
    EnrichmentStatus.PENDING:  {EnrichmentStatus.RUNNING},
    EnrichmentStatus.RUNNING:  {EnrichmentStatus.COMPLETE, EnrichmentStatus.FAILED},
    EnrichmentStatus.FAILED:   {EnrichmentStatus.RUNNING},
    EnrichmentStatus.COMPLETE: set(),                   # terminal
}

VALID_PROJECTION_TRANSITIONS: dict[ProjectionStatus, set[ProjectionStatus]] = {
    ProjectionStatus.PENDING:  {ProjectionStatus.RUNNING},
    ProjectionStatus.RUNNING:  {ProjectionStatus.COMPLETE, ProjectionStatus.FAILED},
    ProjectionStatus.FAILED:   {ProjectionStatus.RUNNING},
    ProjectionStatus.COMPLETE: {ProjectionStatus.RUNNING},  # re-project after enrichment
}


# ── error ─────────────────────────────────────────────────────────────────────

class InvalidTransitionError(ValueError):
    def __init__(self, current: object, next_status: object, status_type: str) -> None:
        super().__init__(
            f"Invalid {status_type} transition: {current!r} -> {next_status!r}"
        )


# ── validators ────────────────────────────────────────────────────────────────

def validate_parse_transition(current: ParseStatus, next_status: ParseStatus) -> None:
    """Raise InvalidTransitionError if the parse transition is not permitted."""
    if next_status not in VALID_PARSE_TRANSITIONS.get(current, set()):
        raise InvalidTransitionError(current, next_status, "parse")


def validate_enrichment_transition(
    current: EnrichmentStatus, next_status: EnrichmentStatus
) -> None:
    """Raise InvalidTransitionError if the enrichment transition is not permitted."""
    if next_status not in VALID_ENRICHMENT_TRANSITIONS.get(current, set()):
        raise InvalidTransitionError(current, next_status, "enrichment")


def validate_projection_transition(
    current: ProjectionStatus, next_status: ProjectionStatus
) -> None:
    """Raise InvalidTransitionError if the projection transition is not permitted."""
    if next_status not in VALID_PROJECTION_TRANSITIONS.get(current, set()):
        raise InvalidTransitionError(current, next_status, "projection")
