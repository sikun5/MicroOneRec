# Historical runtime-script source archive

This directory versions project-owned scripts previously kept only with runtime
artifacts. Each script is an unchanged byte-for-byte copy; `manifest.json` maps
its original repository-relative location and SHA-256. Original runtime files
remain in place. Existing copies of already-versioned source are listed rather
than added again. Data, checkpoints, metrics, logs and environments are excluded.

These are historical snapshots, including debugging and packaging helpers. They
are not active training entrypoints or a promise that every historical experiment
succeeded. Several scripts rely on their original relative location or historical
absolute paths. To reproduce one, restore that original location and its required
runtime inputs; do not recursively execute this archive. The accepted Phase7
training entrypoints remain under `scripts/` and are unchanged by this snapshot.

No experiment was run while making this archive. Migration to the new host and
B-RL remain gated by separate user instructions.
