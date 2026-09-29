# Literature evidence matrix

`literature_evidence_matrix.csv` releases the prior-work coding that supports the mechanism comparison and the
novelty audit of the manuscript (Sections 2–3). It is a structured novelty audit, **not a systematic review**.

**Provenance.** One row per bibliography entry (31). Every coded value is taken verbatim from the manuscript:

| Column(s) | Source in the manuscript |
|---|---|
| `lifecycle_*`, `primary_limitation_lifecycle_table` | Table "Lifecycle comparison of closest prior work" (yes / partial / no) |
| `established_mechanism`, `limitation_derivation_table`, `esaog_design_response` | Table "Literature-to-methodology derivation" |
| `esaog_baseline` | Section 8 (which mechanism-equivalent baseline re-implements the work's mechanism) |
| `design_space_lifecycle_coverage`, `design_space_semantic_formality` | Qualitative positions in the design-space figure (not scores or rankings) |
| `publication_status` | Bibliographic verification against DOI/Crossref, arXiv, W3C, publisher and repository records |

Empty cells mean that the manuscript does not code that work on that dimension; no judgments were added for this
release. The coding dimensions themselves are defined in the manuscript's "Review coding dimensions" table.
