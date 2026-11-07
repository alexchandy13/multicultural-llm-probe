"""Resolve a probe-country CLI argument to the spelling the dataset uses.

Why this exists. A probe run replaces the country named in a prompt with
another country and rescores. For that to measure anything, the substituted
name has to be spelled the way the benchmark spells its own countries --
otherwise the probe prompt and the baseline prompt differ in two ways at once
(the country AND its surface form), and the agreement rate picks up both.

The probe paths used to pass the CLI string straight through, so a job calling
`--probe-country bosnia_and_herzegovina` produced prompts reading
"In bosnia_and_herzegovina," and "country: bosnia_and_herzegovina", while the
US column -- written by the eval scripts, which hardcode a literal -- read
"In the United States,". Every non-US probe column was handicapped against the
US one, on every benchmark.

The dataset is the authority: a name that matches one of its countries
resolves to that country's own spelling, so probe and baseline prompts are
byte-identical apart from the country. CANONICAL only covers probe countries a
given benchmark does not contain at all (BLEnD has 16 countries, and five of
the eight cluster representatives are not among them), where there is no
dataset spelling to defer to.

No torch import, so this is importable and testable without a GPU.
"""

from __future__ import annotations

import re

# Probe countries that some benchmark does not contain, and so has no spelling
# of its own for. Keys are slugs; values are the form to write into a prompt.
# Deliberately not title-casing automatically: "Bosnia And Herzegovina" and
# "the Netherlands" are both wrong in ways a naive rule produces, and an
# unknown name should fail loudly rather than be guessed at.
CANONICAL = {
    "us": "United States",
    "usa": "United States",
    "united_states": "United States",
    "uk": "United Kingdom",
    "united_kingdom": "United Kingdom",
    "netherlands": "Netherlands",
    "hungary": "Hungary",
    "bosnia_and_herzegovina": "Bosnia and Herzegovina",
    "mexico": "Mexico",
    "taiwan": "Taiwan",
    "philippines": "Philippines",
    "iran": "Iran",
    "south_korea": "South Korea",
    "north_korea": "North Korea",
    "hong_kong": "Hong Kong",
    "west_java": "West Java",
    "northern_nigeria": "Northern Nigeria",
    "czech_republic": "Czech Republic",
    "united_arab_emirates": "United Arab Emirates",
    "dominican_republic": "Dominican Republic",
}


def slug(name: str) -> str:
    """Fold a country name to a comparison key: lowercase, spaces as _."""
    return name.strip().lower().replace(" ", "_")


def resolve_probe_country(name: str, dataset_countries) -> str:
    """Return the spelling to substitute for `name` in a probe prompt.

    `dataset_countries` is the benchmark's own set of country values, which
    wins whenever it has a match -- including when its spelling looks odd.
    NormAd stores both "Netherlands" and "hungary", and matching its own
    inconsistency is correct here: the point is that the probe prompt for
    Hungary is spelled exactly like the baseline prompt on Hungary's rows.

    Raises ValueError on a name that neither the dataset nor CANONICAL knows,
    since formatting an unrecognised string into a prompt is the bug this
    function exists to prevent.
    """
    key = slug(name)
    lut = {slug(c): c for c in dataset_countries}
    if key in lut:
        return lut[key]
    if key in CANONICAL:
        return CANONICAL[key]
    raise ValueError(
        f"probe country {name!r} is not one of the benchmark's "
        f"{len(lut)} countries and has no canonical spelling. Add it to "
        f"CANONICAL in evaluate/country_names.py, or pass the dataset's own "
        f"spelling."
    )


# ---------------------------------------------------------------------------
# Country references that are not the country's name
# ---------------------------------------------------------------------------
# CulturalBench-Easy asks its questions in the authors' own words, and 69% of
# the 1,227 never write the country down: 'In Korean dining etiquette, ...',
# 'Which is the preferred way of cooking eggs for Spanish people?', 'What do
# Indians traditionally prefer ...', 'In UK culture, ...'. A probe that only
# replaces the literal name therefore reaches 28% of the benchmark, and the
# 72% it silently leaves alone would be scored as if they had been swapped.
#
# So each country gets the forms it can appear in, and a swap writes the SAME
# form of the target: adjective for adjective, plural for plural. The eight
# probe targets are the entries that have to be exactly right -- they are the
# columns the figures compare against each other -- which is sixteen strings.
# Source-side entries only have to be good enough to find the reference; a
# miss is reported and the row is flagged, never silently left unswapped.
#
# ADJ is the attributive form ('Korean dining etiquette'), PLURAL the bare
# plural used as a noun phrase ('what do Indians prefer'). PLURAL carries its
# own article where English demands one -- 'the Dutch', not 'Dutch' -- because
# the article is part of the noun phrase, not of the sentence around it.
DEMONYMS: dict[str, tuple[str, str]] = {
    # -- the eight probe targets: verify these sixteen strings by eye --------
    "United States":          ("American", "Americans"),
    "Netherlands":            ("Dutch", "the Dutch"),
    "Hungary":                ("Hungarian", "Hungarians"),
    "Bosnia and Herzegovina": ("Bosnian", "Bosnians"),
    "Mexico":                 ("Mexican", "Mexicans"),
    "Taiwan":                 ("Taiwanese", "Taiwanese people"),
    "Philippines":            ("Filipino", "Filipinos"),
    "Iran":                   ("Iranian", "Iranians"),
    # -- source-side only, for detection ------------------------------------
    "United Kingdom":         ("British", "the British"),
    "South Korea":            ("Korean", "Koreans"),
    "North Korea":            ("North Korean", "North Koreans"),
    "China":                  ("Chinese", "Chinese people"),
    "Hong Kong":              ("Hong Kong", "Hong Kongers"),
    "Japan":                  ("Japanese", "Japanese people"),
    "India":                  ("Indian", "Indians"),
    "Spain":                  ("Spanish", "Spaniards"),
    "Morocco":                ("Moroccan", "Moroccans"),
    "Vietnam":                ("Vietnamese", "Vietnamese people"),
    "Malaysia":               ("Malaysian", "Malaysians"),
    "Chile":                  ("Chilean", "Chileans"),
    "Czech Republic":         ("Czech", "Czechs"),
    "United Arab Emirates":   ("Emirati", "Emiratis"),
    "Dominican Republic":     ("Dominican", "Dominicans"),
}

# Alternative names for the same country, as a question might write them.
# Separate from DEMONYMS because these are names, not derived forms: a swap
# replaces an alias with the target's canonical NAME, since the target has no
# alias of its own to use ('In UK culture' -> 'In Hungary culture' is wrong,
# so an alias match on an attributive use resolves to the adjective instead --
# see swap_country_reference).
ALIASES: dict[str, tuple[str, ...]] = {
    "United States":        ("US", "U.S.", "USA", "U.S.A.", "America"),
    "United Kingdom":       ("UK", "U.K.", "Britain", "Great Britain"),
    "South Korea":          ("Korea",),
    "Netherlands":          ("Holland",),
    "United Arab Emirates": ("UAE", "U.A.E."),
    "Czech Republic":       ("Czechia",),
}


# Countries whose name takes "the". Duplicated in eval_culturalbench.py and
# data/prep_culturalbench.py; this is the copy the probe paths use, and the
# other two should import it rather than keep their own.
THE_COUNTRIES = {
    "Netherlands", "United Kingdom", "United States",
    "United Arab Emirates", "Dominican Republic", "Philippines",
    "Czech Republic",
}


def country_with_article(country: str) -> str:
    return f"the {country}" if country in THE_COUNTRIES else country


# Head nouns that make a preceding country reference attributive, where English
# wants the adjective: 'In UK culture' is 'In Hungarian culture', not 'In
# Hungary culture'. Deliberately a closed list of the heads that actually occur
# rather than a rule like "followed by a lowercase word" -- that rule turns
# 'In Japan people often ...' into 'In Hungarian people often ...', because the
# head there belongs to the clause, not to the country.
ATTRIBUTIVE_HEADS = (
    "culture", "cultures", "cultural", "cuisine", "cuisines", "etiquette",
    "society", "societies", "tradition", "traditions", "custom", "customs",
    "household", "households", "family", "families", "dining", "wedding",
    "weddings", "funeral", "funerals", "holiday", "holidays", "context",
)


def _forms_for(country: str) -> list[tuple[str, str]]:
    """Every surface form `country` may appear as, longest first.

    Longest-first matters: 'Taiwanese people' has to be tried before
    'Taiwanese', and 'the Dutch' before 'Dutch', or the shorter form matches
    inside the longer one and the swap mangles what is left behind.
    """
    forms: list[tuple[str, str]] = [("name", country)]
    forms += [("alias", a) for a in ALIASES.get(country, ())]
    adj, plural = DEMONYMS.get(country, (None, None))
    if plural:
        forms.append(("plural", plural))
    if adj:
        forms.append(("adjective", adj))
    return sorted(forms, key=lambda kv: -len(kv[1]))


# (?<!\w)/(?!\w) rather than \b: an alias like "U.S." ends in a non-word
# character, and \b after one demands a word character next, so "In the U.S.,"
# would never match.
def _pattern(kind: str, surface: str) -> str:
    body = re.escape(surface)
    if kind in ("name", "alias"):
        # The article belongs to the matched span, so the replacement decides
        # whether the target takes one -- "the Netherlands" -> "Hungary".
        body = rf"(?:the\s+)?{body}"
    return rf"(?<!\w){body}(?!\w)"


def find_country_reference(text: str, country: str):
    """Locate how `text` refers to `country`.

    Returns (kind, start, end) with kind in name/alias/plural/adjective, or
    None when the question never refers to the country at all -- 846 of
    CulturalBench-Easy's 1,227 name no country, and most of those use a
    demonym, but some genuinely carry the country only as metadata.
    """
    for kind, surface in _forms_for(country):
        m = re.search(_pattern(kind, surface), text)
        if not m:
            continue
        if kind in ("name", "alias"):
            tail = text[m.end():].lstrip()
            head = re.match(r"([A-Za-z]+)", tail)
            if head and head.group(1).lower() in ATTRIBUTIVE_HEADS:
                return "attributive", m.start(), m.end()
        return kind, m.start(), m.end()
    return None


def swap_country_reference(text: str, country: str, target: str):
    """Rewrite `text` to refer to `target` in the same grammatical form.

    Returns (new_text, kind). kind is None when no reference was found, and
    the text comes back unchanged -- the caller flags that row rather than
    scoring it as though a swap had happened, which is how a probe ends up
    silently measuring nothing on 72% of the set.
    """
    hit = find_country_reference(text, country)
    if hit is None:
        return text, None
    kind, start, end = hit

    if target not in DEMONYMS and kind in ("plural", "adjective", "attributive"):
        raise ValueError(
            f"{target!r} is referred to by a demonym here ({kind}), but has no "
            f"DEMONYMS entry in evaluate/country_names.py. Add its adjective "
            f"and plural; guessing them is how a probe column ends up spelled "
            f"differently from the other seven.")

    if kind == "plural":
        repl = DEMONYMS[target][1]
    elif kind in ("adjective", "attributive"):
        repl = DEMONYMS[target][0]
    else:
        repl = country_with_article(target)
    return text[:start] + repl + text[end:], kind
