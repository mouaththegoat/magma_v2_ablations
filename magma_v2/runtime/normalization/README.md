# MAGMA v2 Normalization Layer

The normalization layer separates raw user layout from model-ready layout.

Raw layout is what the user provides: a CSV, a directory, split files, or multiple files that need joining. Normalized layout is the internal contract consumed by `model_worker`.

Examples:

- A CSV with numeric features normalizes to `tabular_dataset`.
- A CSV with `image_path`, `label`, and `patient_id` normalizes to `image_manifest`, not a generic tabular dataset.
- A table with long clinical text normalizes to `text_manifest`.
- A table with ECG/signal paths normalizes to `ecg_manifest`.
- Multiple inputs with join keys can normalize to `multimodal_alignment_manifest`.

`DATA_HANDOFF.json` is the semantic contract between the data worker and model worker. It must explain the observed raw layout, the chosen normalized layout, the prediction unit, label source, split source, path resolution, schema summary, validator reports, assumptions, limitations, and produced files.

File paths alone are not enough. A `.csv` can be a feature table, an image manifest, a text-note table, an ECG manifest, or one side of a multimodal join. Downstream stages must read semantic fields first, then use `train_data`, `val_data`, `test_data`, and `model_ready_data` as the concrete implementation paths.

The data worker should call `normalize_data_inputs` first. The LLM can provide evidence-backed hints such as target or join keys, but the reusable normalization library performs inspection, artifact writing, and validation. The model worker should branch on `DATA_HANDOFF.json["normalized_layout"]` and consume the canonical split files/manifests, not the original raw files.
