# Dataset selection and provenance

The recovery checkpoint selects LandCover.ai v1, original aerial RGB scenes and
dense semantic masks, because dense references allow controlled simulation of
sparse point annotations and dense held-out evaluation.

- Official homepage: https://landcover.ai.linuxpolska.com/
- Original archive: https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip
- Archive SHA-256: dd448287ff3a76291f2b159084b8a265f2ba4559bb80b0551fc8c6c0eb149a4b
- License reported verified in the historical checkpoint: CC BY-NC-SA 4.0.
- Imagery: Head Office of Geodesy and Cartography, Poland.
- Citation: Boguszewski et al., “LandCover.ai: Dataset for Automatic Mapping of
  Buildings, Woodlands, Water and Roads from Aerial Imagery,” CVPR Workshops
  2021, pp. 1102–1110.

Historical inventory: 41 scenes; 33 at 0.25 m/pixel and 8 at 0.5 m/pixel; three
uint8 RGB bands and integer labels 0 background, 1 building, 2 woodland, 3 water,
4 road. These observations must be reverified after access is restored.

The custom geographic split is different from the official prepared 512-pixel
benchmark split. Extraction preserves official split lists. This implementation
uses original scenes and groups before tiling, without resampling.

Limitations include class imbalance, rare land-cover categories, different scene
complexity, unknown shared acquisition identifiers, native-resolution variation
and a geographic scope restricted to the source imagery. Simulated clicks from
dense masks do not measure human annotation time, mistakes or ambiguity. The
repository MIT license does not license imagery or annotations.
