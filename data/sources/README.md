# Data sources

Original data used by Sắc Việt / VietColorNet, one folder per source. These folders are
not stored in Git. Splits built from them live in `data/protocols/`; the processed
ImageNet shards live in `data/processed/`.

| Folder | Content | Used by |
|---|---|---|
| `vicow/` | ViCoW: 1,896 colour/greyscale frame pairs from four Vietnam War-era films (`Color_image/`, `Grayscale_image/`) | protocols V1–V8 (video-disjoint Vietnam test) |
| `vicow_corrected/` | Colour-corrected, sharpened copy of the two ViCoW **training** films (made by `scripts/prepare/correct_vicow_colors.py`; validation/test films stay original in `vicow/`) | protocol V15 |
| `historical_vietnam_images/` | Historical Vietnamese photographs with manual review | 100-image historical audit |
| `viet_cultural_vqa/` | Vietnamese Cultural VQA images and metadata (28,505 images, Hugging Face) | V8–V14 training, V9+ validation/test |
| `ktvic/` | KTVIC life-domain images (train/test Parquet, extracted images) | V8–V14 training, V9+ validation/test |
| `wikimedia_vietnam/` | Licensed Vietnam photographs from Wikimedia Commons | V2+ training |
| `wikimedia_vietnam_military/` | Licensed Vietnam People's Army photographs (`attribution.csv`, `crawl.log`) | V11–V14 training |
| `wikimedia_vietnam_partial/` | Early partial Wikimedia crawl kept for reference | — |
| `imagenet64/` | Downsampled ImageNet 64×64 original batches | `scripts/prepare/prepare_data.py` |

Always preserve the source metadata and verify per-image licenses before publishing or
redistributing a derived dataset.

## Notes

- ViCoW is also available as the original archive
  `ViCoW A Dataset for Colorization and Restoration of Vietnam War Imagery.zip`; the
  folder here uses film names without the `VIDEOn_` prefix, as the protocol CSVs expect.
- Vietnamese Cultural VQA: the Hugging Face git history was removed to save space, so
  re-clone the dataset to update it.
- Wikimedia Commons crawls are resumable, e.g.
  `python scripts/prepare/collect_wikimedia_vietnam.py --output data/sources/wikimedia_vietnam`.
  Wikimedia rate limiting may make this slow.
- UIT-OpenViIC was unavailable on 2026-10-14 (its official Google Drive folder returned
  HTTP 404); no unverified mirror was substituted.
