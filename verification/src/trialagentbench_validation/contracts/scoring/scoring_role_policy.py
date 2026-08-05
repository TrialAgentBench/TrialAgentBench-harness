"""Versioned scoring-role policy for TrialEval route references."""

from __future__ import annotations

from typing import Literal

TrialEvalRouteReferenceRoleV1 = Literal[
    "required_primary",
    "credit_eligible_primary_alternative",
    "sensitivity_only",
    "diagnostic_only",
]
TrialEvalMethodAcceptanceClassV1 = Literal[
    "required_primary", "credit_eligible_primary_alternative", "diagnostic_only"
]

OFFICIAL_SCOREABLE_VARIANT_ROLES_V1 = frozenset(
    {"required_primary", "credit_eligible_primary_alternative"}
)
NON_HEADLINE_VARIANT_ROLES_V1 = frozenset({"sensitivity_only", "diagnostic_only"})
ELIGIBILITY_CLASS_TO_REQUIRED_VARIANT_ROLE_V1 = {
    "required_primary": "required_primary",
    "credit_eligible_primary_alternative": "credit_eligible_primary_alternative",
    "diagnostic_only": "diagnostic_only",
}


def is_official_scoreable_variant_role_v1(role: str) -> bool:
    """Return whether a route-reference role can affect official endpoint scoring."""

    return role in OFFICIAL_SCOREABLE_VARIANT_ROLES_V1


def required_variant_role_for_eligibility_class_v1(eligibility_class: str) -> str:
    """Return the required route-reference role for a method-route eligibility class."""

    try:
        return ELIGIBILITY_CLASS_TO_REQUIRED_VARIANT_ROLE_V1[eligibility_class]
    except KeyError as exc:
        raise ValueError(
            f"Unknown method-route eligibility class: {eligibility_class!r}"
        ) from exc


def validate_method_route_role_alignment_v1(
    *, eligibility_class: str, route_reference_role: str
) -> None:
    """Validate that each method route uses its required scoring role."""

    required_role = required_variant_role_for_eligibility_class_v1(eligibility_class)
    if route_reference_role != required_role:
        raise ValueError(
            f"{eligibility_class!r} method routes require route_reference_role={required_role!r}; "
            f"got {route_reference_role!r}."
        )


__all__ = [
    "ELIGIBILITY_CLASS_TO_REQUIRED_VARIANT_ROLE_V1",
    "NON_HEADLINE_VARIANT_ROLES_V1",
    "OFFICIAL_SCOREABLE_VARIANT_ROLES_V1",
    "TrialEvalMethodAcceptanceClassV1",
    "TrialEvalRouteReferenceRoleV1",
    "is_official_scoreable_variant_role_v1",
    "required_variant_role_for_eligibility_class_v1",
    "validate_method_route_role_alignment_v1",
]
