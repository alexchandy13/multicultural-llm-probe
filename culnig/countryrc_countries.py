"""The CountryRC target country list, kept free of torch and of circular imports.

It lived in scripts/extend_countryrc.py, which imports calc_neuron_score and so
cannot be imported back from it. calc_neuron_score now needs the list too, to run
countryrc on the shardable main pass, so it lives here and both import it.
"""
from __future__ import annotations

# All countries from normad (75 countries) and blend not in the original 8
# TARGET_COUNTRIES (China, Indonesia, Iran, Mexico, South Korea, Spain, UK, USA).
# Display names are inserted directly into CountryRC travel templates.
ALL_EXTRA_COUNTRIES = [
    # normad countries not in original 8
    "Afghanistan", "Argentina", "Australia", "Austria", "Bangladesh",
    "Bosnia and Herzegovina", "Brazil", "Cambodia", "Canada", "Chile",
    "Colombia", "Croatia", "Cyprus", "Egypt", "Ethiopia", "Fiji", "France",
    "Germany", "Greece", "Hong Kong", "Hungary", "India", "Iraq", "Ireland",
    "Israel", "Italy", "Japan", "Kenya", "Laos", "Lebanon", "Malaysia",
    "Malta", "Mauritius", "Myanmar", "Nepal", "Netherlands", "New Zealand",
    "North Macedonia", "Pakistan", "Palestinian Territories",
    "Papua New Guinea", "Peru", "Philippines", "Poland", "Portugal",
    "Romania", "Russia", "Samoa", "Saudi Arabia", "Serbia", "Singapore",
    "Somalia", "South Africa", "South Sudan", "Sri Lanka", "Sudan", "Sweden",
    "Syria", "Taiwan", "Thailand", "Timor-Leste", "Tonga", "Türkiye",
    "Ukraine", "Venezuela", "Vietnam", "Zimbabwe",
    # blend-only (not in normad)
    "Algeria", "Assam", "Nigeria", "North Korea", "West Java", "Azerbaijan",
]
