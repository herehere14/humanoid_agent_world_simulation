"""Shared human-profile assignment for world-sim agents.

These profiles are intentionally more specific than broad temperament labels.
They give each agent a stable bias in how they interpret pressure, protect
themselves, care for others, and narrate events to themselves.
"""

from __future__ import annotations

import random

from .world_agent import Personality


PROFILE_FIELDS = (
    "attachment_style",
    "coping_style",
    "threat_lens",
    "core_need",
    "shame_trigger",
    "care_style",
    "conflict_style",
    "mask_tendency",
    "self_story",
    "longing",
)


DEFAULT_ROLE_PROFILE = {
    "attachment_style": (
        "secure attachment",
        "anxious attachment",
        "guarded attachment",
    ),
    "coping_style": (
        "reach for connection",
        "perform competence",
        "disappear into work",
    ),
    "threat_lens": (
        "chaos",
        "humiliation",
        "abandonment",
    ),
    "core_need": (
        "belonging",
        "control",
        "dignity",
    ),
    "shame_trigger": (
        "looking helpless",
        "being easy to discard",
        "losing face in public",
    ),
    "care_style": (
        "steady presence",
        "practical fixing",
        "quiet encouragement",
    ),
    "conflict_style": (
        "cool negotiation",
        "keep score",
        "appease first",
    ),
    "mask_tendency": (
        "polished competence",
        "dutiful calm",
        "soft warmth",
    ),
    "self_story": (
        "survivor",
        "fixer",
        "witness",
    ),
    "longing": (
        "be safe without owing anyone",
        "be seen clearly",
        "keep one good exit open",
    ),
}


ROLE_HUMAN_PROFILES = {
    "factory_worker": {
        "attachment_style": ("secure attachment", "anxious attachment", "self-protective attachment"),
        "coping_style": ("disappear into work", "deflect with humor", "keep score quietly", "confront head-on"),
        "threat_lens": ("scarcity", "betrayal", "humiliation"),
        "core_need": ("safety", "dignity", "belonging"),
        "shame_trigger": ("failing family", "looking weak on the job", "being talked down to"),
        "care_style": ("protective provisioning", "practical fixing", "steady presence"),
        "conflict_style": ("go sharp", "keep score", "straight negotiation"),
        "mask_tendency": ("dutiful calm", "joke through it", "quiet shutdown"),
        "self_story": ("provider", "loyalist", "survivor"),
        "longing": ("keep the house standing", "be respected without begging", "make it through without bending"),
    },
    "office_professional": {
        "attachment_style": ("guarded attachment", "anxious attachment", "secure attachment"),
        "coping_style": ("perform competence", "intellectualize", "control the room", "disappear into work", "seek witnesses"),
        "threat_lens": ("humiliation", "exposure", "chaos"),
        "core_need": ("dignity", "control", "autonomy"),
        "shame_trigger": ("looking incompetent", "being easy to replace", "needing help in public"),
        "care_style": ("strategic problem-solving", "practical fixing", "quiet encouragement"),
        "conflict_style": ("cool negotiation", "keep score", "command"),
        "mask_tendency": ("polished competence", "emotional shutdown", "command presence"),
        "self_story": ("climber", "fixer", "witness"),
        "longing": ("be taken seriously", "stay impossible to discard", "keep options open"),
    },
    "student": {
        "attachment_style": ("anxious attachment", "secure attachment", "self-protective attachment"),
        "coping_style": ("seek witnesses", "reach for connection", "deflect with humor", "confront head-on"),
        "threat_lens": ("betrayal", "exposure", "abandonment"),
        "core_need": ("justice", "belonging", "truth"),
        "shame_trigger": ("selling out", "being ignored", "looking naive"),
        "care_style": ("strategic problem-solving", "emotional reassurance", "practical fixing"),
        "conflict_style": ("moralize in public", "confront head-on", "triangulate the room"),
        "mask_tendency": ("righteous heat", "joke through it", "soft warmth"),
        "self_story": ("witness", "outsider", "believer"),
        "longing": ("be part of something real", "force the truth into daylight", "not be disposable"),
    },
    "market_vendor": {
        "attachment_style": ("secure attachment", "anxious attachment", "guarded attachment"),
        "coping_style": ("caretake first", "perform competence", "keep score quietly", "reach for connection"),
        "threat_lens": ("scarcity", "betrayal", "chaos"),
        "core_need": ("safety", "usefulness", "belonging"),
        "shame_trigger": ("not being able to provide", "having to ask for mercy", "letting the block down"),
        "care_style": ("protective provisioning", "practical fixing", "steady presence"),
        "conflict_style": ("straight negotiation", "keep score", "appease first"),
        "mask_tendency": ("dutiful calm", "soft warmth", "polished competence"),
        "self_story": ("provider", "guardian", "survivor"),
        "longing": ("make it through the month intact", "keep the stall and the people around it alive", "have one season without scrambling"),
    },
    "dock_worker": {
        "attachment_style": ("secure attachment", "self-protective attachment", "guarded attachment"),
        "coping_style": ("confront head-on", "disappear into work", "keep score quietly"),
        "threat_lens": ("scarcity", "betrayal", "humiliation"),
        "core_need": ("dignity", "belonging", "safety"),
        "shame_trigger": ("failing the family name", "being pushed off the waterfront", "owing the wrong person"),
        "care_style": ("protective provisioning", "steady presence", "practical fixing"),
        "conflict_style": ("go sharp", "straight negotiation", "keep score"),
        "mask_tendency": ("quiet shutdown", "command presence", "dutiful calm"),
        "self_story": ("guardian", "loyalist", "survivor"),
        "longing": ("keep home from being swallowed", "stay rooted where my people know me", "hold the line without selling anyone out"),
    },
    "government_worker": {
        "attachment_style": ("guarded attachment", "secure attachment", "self-protective attachment"),
        "coping_style": ("perform competence", "intellectualize", "control the room"),
        "threat_lens": ("exposure", "humiliation", "chaos"),
        "core_need": ("control", "dignity", "truth"),
        "shame_trigger": ("being publicly tied to the cover-up", "losing bureaucratic control", "looking naive about power"),
        "care_style": ("strategic problem-solving", "quiet encouragement", "practical fixing"),
        "conflict_style": ("command", "cool negotiation", "triangulate the room"),
        "mask_tendency": ("polished competence", "command presence", "emotional shutdown"),
        "self_story": ("operator", "witness", "fixer"),
        "longing": ("get through this without becoming the scapegoat", "keep the paperwork from turning into a verdict", "stay useful to the room that survives"),
    },
    "healthcare": {
        "attachment_style": ("secure attachment", "anxious attachment", "guarded attachment"),
        "coping_style": ("caretake first", "disappear into work", "intellectualize"),
        "threat_lens": ("chaos", "abandonment", "scarcity"),
        "core_need": ("usefulness", "control", "belonging"),
        "shame_trigger": ("failing a patient", "needing rescue while others need me", "freezing when people count on me"),
        "care_style": ("practical fixing", "steady presence", "emotional reassurance"),
        "conflict_style": ("appease first", "cool negotiation", "go sharp"),
        "mask_tendency": ("dutiful calm", "emotional shutdown", "soft warmth"),
        "self_story": ("guardian", "fixer", "witness"),
        "longing": ("keep people alive without losing myself", "get one quiet hour where nobody needs triage", "still be gentle after all this"),
    },
    "community": {
        "attachment_style": ("secure attachment", "anxious attachment", "self-protective attachment"),
        "coping_style": ("reach for connection", "caretake first", "seek witnesses", "perform competence"),
        "threat_lens": ("scarcity", "abandonment", "betrayal"),
        "core_need": ("belonging", "usefulness", "safety"),
        "shame_trigger": ("being unable to help my people", "being treated like I don't matter", "bringing trouble home"),
        "care_style": ("steady presence", "protective provisioning", "emotional reassurance"),
        "conflict_style": ("appease first", "keep score", "moralize in public"),
        "mask_tendency": ("soft warmth", "dutiful calm", "joke through it"),
        "self_story": ("guardian", "story carrier", "survivor"),
        "longing": ("keep my people connected", "be held without having to earn it", "make the neighborhood feel survivable again"),
    },
}


def _assign_defaults(personality: Personality, pool: dict[str, tuple[str, ...]], rng: random.Random) -> dict[str, str]:
    assigned: dict[str, str] = {}
    for field in PROFILE_FIELDS:
        explicit = getattr(personality, field, "")
        if explicit:
            assigned[field] = explicit
        else:
            options = pool.get(field) or DEFAULT_ROLE_PROFILE[field]
            assigned[field] = rng.choice(options)
    return assigned


def assign_human_profile(
    personality: Personality,
    role: str,
    rng: random.Random,
    *,
    identity_tags: tuple[str, ...] = (),
    private_burden: str = "",
) -> None:
    """Fill in human-profile fields for a personality if they are unset."""
    pool = ROLE_HUMAN_PROFILES.get(role, DEFAULT_ROLE_PROFILE)
    assigned = _assign_defaults(personality, pool, rng)

    tags_text = " ".join(identity_tags).lower()
    burden = private_burden.lower()

    if (
        any(token in tags_text for token in ("provider", "rent stressed", "cash flow watcher", "survival math", "family crew"))
        or any(token in burden for token in ("bill", "rent", "lease", "grocery", "landlord", "boat payment", "tuition", "loan"))
    ):
        assigned["threat_lens"] = "scarcity"
        assigned["core_need"] = "safety"
        assigned["shame_trigger"] = "failing family"
        if assigned["self_story"] not in {"provider", "guardian"}:
            assigned["self_story"] = "provider"

    if any(token in tags_text for token in ("message discipline", "paper trail keeper", "procedural loyalist", "knows the numbers", "polished operator")):
        assigned["coping_style"] = "perform competence"
        assigned["mask_tendency"] = "polished competence"

    if any(token in tags_text for token in ("organizer", "document hoarder", "story carrier", "story chaser", "network builder", "phone tree runner")):
        assigned["coping_style"] = "seek witnesses"
        assigned["core_need"] = "truth" if any(token in tags_text for token in ("story", "document")) else "justice"
        assigned["self_story"] = "witness"

    if any(token in tags_text for token in ("triage first", "community kitchen", "shelter volunteer", "mutual aid", "company heart")):
        assigned["coping_style"] = "caretake first"
        assigned["care_style"] = "practical fixing"
        assigned["core_need"] = "usefulness"
        assigned["mask_tendency"] = "dutiful calm"

    if any(token in tags_text for token in ("territorial", "harbor loyalist", "neighborhood rooted", "neighborhood anchor")):
        assigned["threat_lens"] = "scarcity"
        assigned["self_story"] = "guardian"

    if burden and any(token in burden for token in ("hide", "hidden", "deleted", "screenshots", "kept quiet", "copied", "waiver")):
        assigned["threat_lens"] = "exposure"

    if burden and any(token in burden for token in ("source", "favor", "buyer", "promised", "bridge loan", "safety shortcut")):
        assigned["conflict_style"] = "keep score"

    if any(token in tags_text for token in ("career climber", "deal facing", "polished operator")):
        assigned["self_story"] = "climber"
        if assigned["core_need"] == "safety":
            assigned["core_need"] = "dignity"

    for field, value in assigned.items():
        setattr(personality, field, value)
