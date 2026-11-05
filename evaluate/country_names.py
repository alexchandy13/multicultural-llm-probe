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
