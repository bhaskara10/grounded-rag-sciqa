"""
Tests for sciqa_schema state machine transitions.

Validates:
- all valid transitions are accepted (no error)
- all invalid transitions raise InvalidTransitionError
- parse / enrichment are terminal at COMPLETE
- projection is NOT terminal at COMPLETE (re-projection after enrichment)
- failed states allow retry (-> running)
"""
import pytest
from sciqa_schema.enums import EnrichmentStatus, ParseStatus, ProjectionStatus
from sciqa_schema.transitions import (
    InvalidTransitionError,
    validate_enrichment_transition,
    validate_parse_transition,
    validate_projection_transition,
)


class TestParseTransitions:
    def test_pending_to_running(self):
        validate_parse_transition(ParseStatus.PENDING, ParseStatus.RUNNING)

    def test_running_to_complete(self):
        validate_parse_transition(ParseStatus.RUNNING, ParseStatus.COMPLETE)

    def test_running_to_failed(self):
        validate_parse_transition(ParseStatus.RUNNING, ParseStatus.FAILED)

    def test_failed_to_running_retry_allowed(self):
        validate_parse_transition(ParseStatus.FAILED, ParseStatus.RUNNING)

    def test_complete_is_terminal(self):
        with pytest.raises(InvalidTransitionError):
            validate_parse_transition(ParseStatus.COMPLETE, ParseStatus.RUNNING)

    def test_complete_to_pending_forbidden(self):
        with pytest.raises(InvalidTransitionError):
            validate_parse_transition(ParseStatus.COMPLETE, ParseStatus.PENDING)

    def test_pending_to_complete_skip_running_forbidden(self):
        with pytest.raises(InvalidTransitionError):
            validate_parse_transition(ParseStatus.PENDING, ParseStatus.COMPLETE)

    def test_pending_to_failed_skip_running_forbidden(self):
        with pytest.raises(InvalidTransitionError):
            validate_parse_transition(ParseStatus.PENDING, ParseStatus.FAILED)


class TestEnrichmentTransitions:
    def test_pending_to_running(self):
        validate_enrichment_transition(EnrichmentStatus.PENDING, EnrichmentStatus.RUNNING)

    def test_running_to_complete(self):
        validate_enrichment_transition(EnrichmentStatus.RUNNING, EnrichmentStatus.COMPLETE)

    def test_running_to_failed(self):
        validate_enrichment_transition(EnrichmentStatus.RUNNING, EnrichmentStatus.FAILED)

    def test_failed_to_running_retry_allowed(self):
        validate_enrichment_transition(EnrichmentStatus.FAILED, EnrichmentStatus.RUNNING)

    def test_complete_is_terminal(self):
        with pytest.raises(InvalidTransitionError):
            validate_enrichment_transition(EnrichmentStatus.COMPLETE, EnrichmentStatus.RUNNING)

    def test_complete_to_pending_forbidden(self):
        with pytest.raises(InvalidTransitionError):
            validate_enrichment_transition(EnrichmentStatus.COMPLETE, EnrichmentStatus.PENDING)


class TestProjectionTransitions:
    def test_pending_to_running(self):
        validate_projection_transition(ProjectionStatus.PENDING, ProjectionStatus.RUNNING)

    def test_running_to_complete(self):
        validate_projection_transition(ProjectionStatus.RUNNING, ProjectionStatus.COMPLETE)

    def test_running_to_failed(self):
        validate_projection_transition(ProjectionStatus.RUNNING, ProjectionStatus.FAILED)

    def test_failed_to_running_retry_allowed(self):
        validate_projection_transition(ProjectionStatus.FAILED, ProjectionStatus.RUNNING)

    def test_complete_to_running_allowed_for_reprojection(self):
        # projection is NOT terminal — re-runs after enrichment completes
        validate_projection_transition(ProjectionStatus.COMPLETE, ProjectionStatus.RUNNING)

    def test_complete_to_pending_forbidden(self):
        with pytest.raises(InvalidTransitionError):
            validate_projection_transition(ProjectionStatus.COMPLETE, ProjectionStatus.PENDING)

    def test_complete_to_failed_forbidden(self):
        with pytest.raises(InvalidTransitionError):
            validate_projection_transition(ProjectionStatus.COMPLETE, ProjectionStatus.FAILED)
