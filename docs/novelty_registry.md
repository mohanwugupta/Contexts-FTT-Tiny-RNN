# Frozen graded contextual novelty

The registry `configs/novelty_registry_v1.json` was created after geometric tests and before any checkpoint was evaluated on its new cues. Its SHA256 is calculated over canonical JSON excluding the self-referential `registry_sha256` field. Evaluation verifies both this digest and the exact prospective geometry.

A=(1,0,0), B=(0,1,0); Mixed=(.5,.5,0), NoContext=(0,0,0). Novel30_A=(sqrt(3)/2,-1/2,0), Novel30_B=(-1/2,sqrt(3)/2,0), Novel60_A=(1/2,-sqrt(3)/2,0), Novel60_B=(-sqrt(3)/2,1/2,0). Each Novel is unit norm, outside the familiar angular sector and A/B convex segment. The nearest-endpoint angular novelty is 30 or 60 degrees. This operationalizes continuous out-of-support generalization, not a new semantic context dimension.

Novel30Mean and Novel60Mean each average their two raw directions within seed. NovelMean averages all four raw variants. Historical Novel1–4 are supplementary LegacyNovel1–4 and never enter these aggregates.

Prospectively, the contrast coordinate maps same-side 30 and 60 degree cues to the same alpha (approximately ±1.3660254); it discards their differing orthogonal residuals. These residuals are measured, never supplied to a new model pathway. No monotonic behavioral effect is required.
