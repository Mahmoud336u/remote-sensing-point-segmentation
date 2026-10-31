# Recovery and reconstruction

The user supplied a recovery checkpoint dated 2026-10-31. Inspection found only
README.md, LICENSE and .gitignore at base commit
1f77b1099a5be1cb110ea4a5b5c414121d856eab. The sole remote branch was main at that
commit. A search of the accessible /workspace and /tmp file trees found no
restored implementation, raw archive, split manifest or trained checkpoint.
The initial checkout was clean and was preserved outside the repository.

The implementation in this branch was reconstructed from behavioral contracts.
No original source fingerprint is claimed to match. In particular:

- The original seed algorithm was not recovered. The new algorithm uses the
  first eight bytes of SHA-256 over compact JSON
  [seed-version, run-seed, source-ID, tile-ID, namespace], interpreted big endian.
  Its version is sha256-json-v1-reconstructed. Uniform budgets share a namespace;
  stratified sampling and each augmentation epoch use separate namespaces.
- The original manifest was not recovered. Group ordering, RNG assignment,
  serialization and subset selection are now explicitly implemented. Any newly
  generated manifest gets a new hash and a reconstruction label. Historical
  25/6/6 group counts and distance values remain reference observations only.
- The subset algorithm is a class-blind PCG64 permutation over sorted windows.
  It is independent of condition and run seed. It does not promise the historical
  smoke's scene composition.
- The lost checkpoint cannot be recreated from its hash. New weights require
  retraining. Hashes of reconstructed results must be recorded separately.
- Tests were authored for these reconstructed contracts. Historical test names
  and totals do not imply the original tests were recovered.

The original data license and publication facts are from the user-supplied
checkpoint until the official source can be reached again. The official
ResNet-18 artifact was independently downloaded and its full hash verified here.
See recovery_checkpoint.json for historical provenance and
results/verification for new evidence.

No environment was published or reconfigured during recovery. The local setup
script prepares this checkout and does not modify cloud environment settings.
