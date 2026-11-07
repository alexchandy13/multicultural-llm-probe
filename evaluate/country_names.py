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
# Each country maps to its forms. The FIRST entry of each tuple is what gets
# written when this country is the probe TARGET; every entry is matched when it
# is the SOURCE. One country needs several detection forms -- CulturalBench
# writes both "In Philippine culture" and "Filipino", and "Nepali" and
# "Nepalese" -- but it must write exactly one form as a target, or the eight
# probe columns differ in surface form as well as in country, which is the bug
# this whole module exists to prevent.
#
# "English" is deliberately absent from United Kingdom: it is also the language,
# and "Do people speak English?" would match and be swapped into nonsense.
DEMONYMS: dict[str, dict[str, tuple[str, ...]]] = {
    # -- the eight probe targets: check the first string of each tuple --------
    "United States":          {"adj": ("American",),     "plural": ("Americans",)},
    "Netherlands":            {"adj": ("Dutch",),        "plural": ("the Dutch",)},
    "Hungary":                {"adj": ("Hungarian",),    "plural": ("Hungarians",)},
    "Bosnia and Herzegovina": {"adj": ("Bosnian",),      "plural": ("Bosnians",)},
    "Mexico":                 {"adj": ("Mexican",),      "plural": ("Mexicans",)},
    "Taiwan":                 {"adj": ("Taiwanese",),    "plural": ("Taiwanese people",)},
    "Philippines":            {"adj": ("Filipino", "Philippine", "Filipina"),
                               "plural": ("Filipinos", "Filipinas")},
    "Iran":                   {"adj": ("Iranian", "Persian"),
                               "plural": ("Iranians", "Persians")},
    # -- the remaining CulturalBench-Easy countries, detection and targets ---
    "Argentina":              {"adj": ("Argentine", "Argentinian"),
                               "plural": ("Argentines", "Argentinians")},
    "Australia":              {"adj": ("Australian", "Aussie"),
                               "plural": ("Australians", "Aussies")},
    "Bangladesh":             {"adj": ("Bangladeshi", "Bengali", "Bangali"),
                               "plural": ("Bangladeshis", "Bengalis")},
    "Brazil":                 {"adj": ("Brazilian",),    "plural": ("Brazilians",)},
    "Canada":                 {"adj": ("Canadian",),     "plural": ("Canadians",)},
    "Chile":                  {"adj": ("Chilean",),      "plural": ("Chileans",)},
    "China":                  {"adj": ("Chinese",),      "plural": ("Chinese people",)},
    "Czech Republic":         {"adj": ("Czech",),        "plural": ("Czechs",)},
    "Egypt":                  {"adj": ("Egyptian",),     "plural": ("Egyptians",)},
    "France":                 {"adj": ("French",),       "plural": ("the French",)},
    "Germany":                {"adj": ("German",),       "plural": ("Germans",)},
    "Hong Kong":              {"adj": ("Hong Kong", "Hongkong"),
                               "plural": ("Hong Kongers", "Hongkongers")},
    "India":                  {"adj": ("Indian",),       "plural": ("Indians",)},
    "Indonesia":              {"adj": ("Indonesian",),   "plural": ("Indonesians",)},
    "Israel":                 {"adj": ("Israeli",),      "plural": ("Israelis",)},
    "Italy":                  {"adj": ("Italian",),      "plural": ("Italians",)},
    "Japan":                  {"adj": ("Japanese",),     "plural": ("Japanese people",)},
    "Lebanon":                {"adj": ("Lebanese",),     "plural": ("the Lebanese",)},
    "Malaysia":               {"adj": ("Malaysian",),    "plural": ("Malaysians",)},
    "Morocco":                {"adj": ("Moroccan",),     "plural": ("Moroccans",)},
    "Nepal":                  {"adj": ("Nepali", "Nepalese"),
                               "plural": ("Nepalis", "Nepalese people")},
    "New Zealand":            {"adj": ("New Zealand", "Kiwi"),
                               "plural": ("New Zealanders", "Kiwis")},
    "Nigeria":                {"adj": ("Nigerian",),     "plural": ("Nigerians",)},
    "North Korea":            {"adj": ("North Korean",), "plural": ("North Koreans",)},
    "Pakistan":               {"adj": ("Pakistani",),    "plural": ("Pakistanis",)},
    "Peru":                   {"adj": ("Peruvian",),     "plural": ("Peruvians",)},
    "Poland":                 {"adj": ("Polish",),       "plural": ("Poles",)},
    "Romania":                {"adj": ("Romanian",),     "plural": ("Romanians",)},
    "Russia":                 {"adj": ("Russian",),      "plural": ("Russians",)},
    "Saudi Arabia":           {"adj": ("Saudi", "Saudi Arabian"),
                               "plural": ("Saudis", "Saudi Arabians")},
    "Singapore":              {"adj": ("Singaporean",),  "plural": ("Singaporeans",)},
    "South Africa":           {"adj": ("South African",), "plural": ("South Africans",)},
    "South Korea":            {"adj": ("Korean",),       "plural": ("Koreans",)},
    "Spain":                  {"adj": ("Spanish",),      "plural": ("Spaniards",)},
    "Thailand":               {"adj": ("Thai",),         "plural": ("Thais",)},
    "Turkey":                 {"adj": ("Turkish",),      "plural": ("Turks",)},
    "Ukraine":                {"adj": ("Ukrainian",),    "plural": ("Ukrainians",)},
    "United Arab Emirates":   {"adj": ("Emirati",),      "plural": ("Emiratis",)},
    "United Kingdom":         {"adj": ("British",),
                               "plural": ("the British", "Brits", "Britons")},
    "Vietnam":                {"adj": ("Vietnamese",),   "plural": ("Vietnamese people",)},
    "Zimbabwe":               {"adj": ("Zimbabwean",),   "plural": ("Zimbabweans",)},
    "Dominican Republic":     {"adj": ("Dominican",),    "plural": ("Dominicans",)},
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
    "Turkey":               ("Türkiye", "Turkiye"),
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
    entry = DEMONYMS.get(country, {})
    forms += [("plural", p) for p in entry.get("plural", ())]
    forms += [("adjective", a) for a in entry.get("adj", ())]
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
        repl = DEMONYMS[target]["plural"][0]
    elif kind in ("adjective", "attributive"):
        repl = DEMONYMS[target]["adj"][0]
    else:
        repl = country_with_article(target)
    return text[:start] + repl + text[end:], kind


def strip_country_reference(text: str, country: str) -> tuple[str, bool]:
    """Delete `text`'s reference to `country`, returning (new_text, applied).

    adjective / attributive: dropping just the word stays grammatical, because
    it is a modifier -- 'In Korean dining etiquette' -> 'In dining etiquette',
    'for Spanish people' -> 'for people'. This is the big win: 725 of
    CulturalBench-Easy's 1,227 rows use one of these two forms.

    name / alias: dropped together with an immediately preceding preposition
    ('in'/'from') and a following comma, or 'In the Netherlands, which...'
    becomes 'In, which...' instead of 'Which...'.

    plural: 'What do Indians prefer' and 'the communication style of the
    Dutch' use the demonym as the sentence's subject or object, not as a
    modifier, so deleting it outright would leave no noun phrase at all
    ('What do prefer'). Substituted with 'people' instead -- 'What do people
    prefer', 'the communication style of people' -- which keeps the sentence
    about a culture in general rather than inventing a specific one.

    No reference found: returned unchanged; the caller must flag the row as
    not run through the no-country arm.
    """
    hit = find_country_reference(text, country)
    if hit is None:
        return text, False
    kind, start, end = hit

    if kind == "plural":
        repl = "People" if start == 0 else "people"
        return _tidy_spacing(text[:start] + repl + text[end:]), True

    clause = _delete_intro_clause(text, start)
    if clause is not None:
        return clause, True

    if kind in ("adjective", "attributive"):
        return _delete_span(text, start, end), True

    # name / alias
    return _delete_prepositional_phrase(text, start, end), True


# Clause-opening words that can introduce a droppable scene-setting clause:
# 'In Egyptian culture, when is...', 'Among Valencian people, how is...',
# 'During a traditional Yoruba gathering, how do...'. A closed list rather
# than "any preposition", the same policy as ATTRIBUTIVE_HEADS -- 'for
# Spanish people' is not an intro clause, it's the whole question, and
# dropping through ITS nearest comma (if any) would eat part of the question.
_CLAUSE_OPENERS = re.compile(
    r"^(?:in|from|among|within|throughout|across|during)\b", re.I)


def _delete_intro_clause(text: str, ref_start: int) -> str | None:
    """Drop a whole sentence-initial scene-setting clause, through its first
    comma, when the country/cultural reference sits inside it. Returns None
    when the text isn't shaped that way, so the caller falls back to deleting
    just the matched word.

    This exists because deleting only the adjective inside a multi-word head
    noun phrase breaks the sentence -- 'In Egyptian culture,' has 'culture'
    as its real head, 'Korean dining etiquette' has 'etiquette' two words
    later, and either way 'In culture,' / 'In etiquette,' is left standing on
    its own as a sentence opener, which is not a thing English does. A comma-
    delimited introductory clause, whatever words it contains, is always
    droppable: the clause after the comma was written to stand as a complete
    question on its own, so cutting there needs no parse of what came before.
    """
    if not _CLAUSE_OPENERS.match(text):
        return None
    comma = text.find(",")
    if comma == -1 or ref_start >= comma:
        return None
    rest = text[comma + 1 :].lstrip()
    if rest:
        rest = rest[0].upper() + rest[1:]
    return _tidy_spacing(rest)


def _delete_span(text: str, start: int, end: int) -> str:
    """Delete text[start:end], the plain case: the word was a modifier."""
    return _tidy_spacing(text[:start] + text[end:])


def _delete_prepositional_phrase(text: str, start: int, end: int) -> str:
    """Delete text[start:end] along with a preceding 'in'/'from' and a
    following comma, so 'In the Netherlands, which...' becomes 'Which...'
    rather than 'In, which...'. Shared by country names/aliases and by
    unregistered name-form phrases like 'Judaism' or 'rainbow nation' -- see
    strip_unregistered_reference().
    """
    pre, tail = text[:start], text[end:]
    m = re.search(r"\b(?:in|from)\s*$", pre, re.I)
    new_pre = pre[: m.start()] if m else pre
    sentence_start = bool(m) and not new_pre.strip()
    new_tail = re.sub(r"^\s*,\s*", "" if sentence_start else " ", tail)
    new = _tidy_spacing(new_pre + new_tail)
    if sentence_start and new:
        new = new[0].upper() + new[1:]
    return new


# ---------------------------------------------------------------------------
# References CulturalBench-Easy makes that are not to a country at all
# ---------------------------------------------------------------------------
# 135 of the 1,227 questions name no country -- not even a demonym -- because
# they refer to an ethnic group ('a young Yoruba man'), a religion ('In
# Judaism,'), or a sub-national region ('many South American cultures')
# instead. These rows are never candidates for the country-swap probe: there
# is no country-level target to substitute Yoruba or Judaism FOR. But the
# no-country arm only needs to delete the reference, not replace it with
# anything, and that is just as safe to do here as for a registered country's
# adjective -- 'a young Yoruba man' -> 'a young man' is exactly the same move
# as 'Korean dining etiquette' -> 'dining etiquette'.
#
# A closed, by-hand list rather than a part-of-speech heuristic, for the same
# reason DEMONYMS and ATTRIBUTIVE_HEADS are closed lists: a rule that deletes
# "any capitalized word before a head noun" would also delete "English" from
# "an English-language form" or "Western" from "a Western diplomat", whatever
# those turn out to mean in context neither of us has read.
UNREGISTERED_ADJECTIVES = (
    "Yoruba", "Igbo", "Zulu", "Xhosa", "Batswana", "Chichewa",
    "Shona", "Mashona", "Sahraoui", "Saharoui", "Valencian",
    "Scottish", "Jewish", "Hindu", "Arabic", "Slavic", "South American",
)

# Proper nouns that function like a country name in an 'In X,'/'from X' clause
# but name a religion or a nickname rather than a country.
UNREGISTERED_NAME_PHRASES = (
    "Judaism", "Sikhism", "rainbow nation",
)


def strip_unregistered_reference(text: str) -> tuple[str, bool]:
    """Delete a non-country cultural reference strip_country_reference can't
    see, because it was never told the row's country at all -- this is the
    fallback for the 135 CulturalBench-Easy rows find_country_reference
    returns None on. Tries phrases before single words, since 'rainbow
    nation' must not be matched by a search for 'nation' alone (it isn't in
    the word list; this comment is the reminder for the next word added).
    """
    for phrase in UNREGISTERED_NAME_PHRASES:
        m = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.I)
        if m:
            clause = _delete_intro_clause(text, m.start())
            if clause is not None:
                return clause, True
            return _delete_prepositional_phrase(text, m.start(), m.end()), True
    for word in UNREGISTERED_ADJECTIVES:
        m = re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text)
        if m:
            clause = _delete_intro_clause(text, m.start())
            if clause is not None:
                return clause, True
            return _delete_span(text, m.start(), m.end()), True
    return text, False


def _tidy_spacing(s: str) -> str:
    """Collapse runs of spaces and remove a space left before punctuation.

    'aspect in Nigeria?' loses 'in Nigeria' and leaves 'aspect ?' -- the
    deleted span bordered the question mark on one side with nothing on the
    other, so there is no following word to normally absorb the space.
    """
    s = re.sub(r" {2,}", " ", s).strip()
    return re.sub(r"\s+([?.,!:;])", r"\1", s)
