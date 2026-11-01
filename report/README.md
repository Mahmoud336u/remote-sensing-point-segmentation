# Technical assessment report and evidence

The Phase 5 package is prepared for review, but the empirical assessment remains
incomplete. There are zero real-data campaign runs. The report contains methods,
verification and a blocked-results table; it does not invent performance figures.

- technical_report.pdf and technical_report.md: rendered report and editable source.
- reviewer_audit.md: original requirements checklist and justified 65/100 score.
- evidence_inventory.json / .csv: claims, source hashes and every candidate run.
- figures/workflow.dot / .pdf / .png: editable conceptual workflow; not a result chart.
- reference_checks.json: bibliography/source retrieval scope.

Rebuild the PDF from the repository root with `bash report/build_report.sh`.
This uses Pandoc, XeLaTeX, Graphviz and DejaVu fonts (system tools, separate from
the Python lock). Validated versions are recorded in
results/verification/report_render.json. A reader does not need these tools to
open the supplied PDF.

After any source changes, run `python -m scripts.verify_tests`, execute the
notebook, then `python -m scripts.audit_evidence`. Build the reviewed package with
`python -m scripts.build_submission`; verify an existing ZIP using
`python -m scripts.build_submission --verify-only`. The ZIP's internal manifest
contains per-file SHA-256 digests. The external .zip.sha256 covers the archive.
