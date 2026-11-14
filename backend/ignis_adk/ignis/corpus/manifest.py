"""Verified NASA NTRS record identifiers used by the IGNIS literature corpus.

Every ID below was retrieved live from the public NTRS search API on
2026-11-10/11. Downloads are fetched from the official NTRS API
(https://ntrs.nasa.gov/api/citations/{id}/downloads/...). Records that do not
provide a public PDF are skipped and reported — never substituted.

`--refresh` now UNIONS freshly-discovered IDs with this hand-verified base
manifest so re-ingestion never silently loses curated records.
"""

INVESTIGATION_QUERIES = {
    # Core campaigns (live-verified 2026-11-10)
    "BASS": "Burning and Suppression of Solids",
    "BASS-II": "BASS-II Burning and Suppression of Solids",
    "FLEX": "Flame Extinguishment Experiment",
    "ACME": "Advanced Combustion via Microgravity Experiments",
    "SAME": "Smoke Aerosol Measurement Experiment",
    "SAME-R": "Smoke Aerosol Measurement Experiment reflight",
    "SAFFIRE": "SAFFIRE Spacecraft Fire Experiment",
    # Expanded coverage ( combustion / fire safety in microgravity )
    "CIR": "Combustion Integrated Rack",
    "FLAME-SPREAD": "flame spread microgravity",
    "FLAMMABILITY": "microgravity flammability",
    "COOL-FLAMES": "cool flame microgravity combustion",
    "SMOKE-DET": "smoke detector ISS spacecraft",
    "MAT-FLAM": "material flammability spacecraft",
    "FIRE-SAFETY": "spacecraft fire safety",
}

# Hand-picked, live-verified record IDs (see scripts/ingest_corpus.py --refresh
# to rebuild this list from the current NTRS API).
MANIFEST: dict[str, list[int]] = {
    "BASS-II": [20210011385, 20160000593, 20170006615],
    "FLEX": [20090014812, 20150023456, 20260001908],
    "ACME": [20210015913, 20220015583, 20210021781, 20210022542],  # 20210022542 = JASMAC-33 flames paper
    "SAME": [20130000422, 20170007283, 20070031686],
    "SAFFIRE": [20170000230, 20170001636, 20220002714],
}

NTRS_SEARCH = "https://ntrs.nasa.gov/api/citations/search"
NTRS_RECORD = "https://ntrs.nasa.gov/api/citations/{id}"
NTRS_CITATION = "https://ntrs.nasa.gov/citations/{id}"
USER_AGENT = "IGNIS-SpaceCombustionIntelligence/1.0 (corpus ingestion; research use)"
