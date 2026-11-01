# Original assessment requirements

Source: user-uploaded **Technical Assessment.pdf**, one page. The PDF was read by
text extraction and visual inspection, including the embedded problem statement.
Its SHA-256 and exact task text are recorded in [assessment_source.json](assessment_source.json).
The original document is retained in the session attachment storage and is not
redistributed in the submission or licensed by this repository's MIT license.

1. “Implement partial Cross entropy loss”
2. “Find any remote sensing image segmentation data, randomly sample the simulated
   point label, and add the loss to any remote sensing segmentation network”
3. “Design experiments, explore one or two factors that affect the performance,
   and write technical reports (method + experiment(purpose/hypothesis +
   experimental process + results))”

The tools paragraph requires Python, permits any deep learning toolbox and accepts
Python files or a Jupyter notebook with supporting documentation.

The introductory image discusses transfer learning, a partial loss, semi-supervised
learning, test augmentation, ensembles and histogram matching as background to a
landvisor framework. These are context in the supplied document, not additional
instructions overriding the user's Phase 5 request. The explicit task asks for
Partial Cross-Entropy. Its implementation follows ordinary cross-entropy on the
selected labels, as the user's mathematical contract specifies; it does not add
the focal-loss weighting visible in the illustrative background equation.

The exact requirement-to-evidence assessment is in
[reviewer_audit.md](../report/reviewer_audit.md). Task 1 is verified on current
software tests. Tasks 2 and 3 remain incomplete on real data and measured results.
