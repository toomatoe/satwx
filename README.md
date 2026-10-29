# satwx: weather-pattern probabilities from GOES satellite imagery

`classify` takes a place and time (or an IR image tile) and returns a calibrated probability for each of nine
weather-pattern classes. It reads GOES-19/18 ABI infrared imagery straight from NOAA's open-data bucket on
AWS, with no account or API key.

```
$ classify --lat 17.5 --lon -78.1 --time 2025-10-28T12:00      # Hurricane Melissa
      satellite: goes19
     scan_start: 2025-10-28T12:00:20
          model: satwx_ir4.pt
 tropical_cyclone  0.996 ########################################
 convective_storm  0.004
            other  0.000
            ...
```

Classes: `clear`, `cumulus`, `stratiform`, `low_stratus_fog`, `cirrus`, `convective_storm`,
`tropical_cyclone`, `frontal_band`, `other` (mixed or ambiguous scenes that match no rule).

See [docs/FEASIBILITY.md](docs/FEASIBILITY.md) for the research behind the design: data sources, available
labelled datasets and the state of the art.

## How it works

1. **Imagery.** GOES-19 (GOES-East) full-disk ABI L2 cloud and moisture imagery: four IR bands (C07 3.9 µm,
   C08 6.2 µm, C13 10.3 µm, C15 12.3 µm) as brightness temperatures. These are available day and night. The
   disk is cut into 128×128-pixel tiles at 2 km (about 256 km across), mean-pooled to 64×64.
2. **Weak labels.** No open dataset labels these classes ([details](docs/FEASIBILITY.md#2-labelled-datasets)),
   so each tile is labelled by rules over *independent* products the model never sees
   ([satwx/labels.py](satwx/labels.py)):
   | class | rule (first match wins, in this order) |
   |---|---|
   | tropical_cyclone | tile centre ≤ 200 km from an IBTrACS tropical storm (≥ 34 kt), cloud fraction > 0.5 |
   | convective_storm | ≥ 10 GLM flashes in the tile during the scan and > 2 % of pixels colder than 235 K |
   | frontal_band | a WPC-analysed surface front crosses the tile, cloud fraction > 0.6 |
   | clear | ABI clear-sky-mask cloud fraction < 3 % |
   | low_stratus_fog | cloud fraction > 85 %, > 90 % liquid tops, median cloud-top height < 1.5 km |
   | cirrus | cloud fraction > 50 %, > 80 % ice tops, median top > 7 km, few cold (< 235 K) pixels |
   | stratiform | cloud fraction > 85 %, uniform tops (C13 std < 5 K), no deep cold tops |
   | cumulus | cloud fraction 15–70 %, liquid, tops < 3.5 km, no cold tops |
   | other | anything else, including tiles with a little lightning |
3. **Model.** ImageNet-pretrained ResNet-18 with its first convolution adapted to the IR channels, fine-tuned
   for 6 epochs on CPU (a 4-band model and a C13-only model) ([scripts/train.py](scripts/train.py)). Storm-centred tiles are oversampled for the
   rare TC class. **Temperature scaling** is fitted on validation scenes, so the probabilities are calibrated.
4. **Split.** The dataset runs from June 2025 to May 2026 (92 regular scenes every 4 days, plus 84 extra
   TC-centred scenes). It is cut into 16-day blocks cycling train, train, train, val, test. Training scenes
   within 4 days of a held-out block are dropped, and training TC scenes within 12 days. So no storm and no
   synoptic regime appears in both training and evaluation.

A second model, `satwx_ir1.pt`, uses only C13. That lets `classify image.png` work on a single greyscale IR
image.

## Results (held-out test blocks, regular tile grid)

The test set is 39 GOES-19 scenes from 5 held-out 16-day blocks: Aug 4–19 2025, Oct 23–Nov 7 2025,
Jan 2026 and Apr 2026. It holds 18,448 tiles on the regular grid plus 339 storm-centred tiles from 6 storms
(Erin, Melissa, Dexter, Henriette, Ivo, Sonia). None of these storms appear in training. The full numbers
are in `models/*.metrics.json`.

| model | accuracy | macro-F1 | NLL | ECE (15 bins) | majority-class baseline |
|---|---|---|---|---|---|
| `satwx_ir4.pt` (C07, C08, C13, C15) | **0.820** | 0.638 | 0.446 | 0.008 | 0.563 |
| `satwx_ir1.pt` (C13 only, used for PNG) | 0.782 | 0.587 | 0.528 | 0.006 | 0.563 |

Per class, for `satwx_ir4.pt` on test tiles. *mean P* is the average predicted probability, against the class's
actual frequency:

| class | precision | recall | F1 | support | freq | mean P |
|---|---|---|---|---|---|---|
| clear | 0.84 | 0.83 | 0.83 | 506 | 0.027 | 0.029 |
| cumulus | 0.82 | 0.84 | 0.83 | 2727 | 0.148 | 0.151 |
| stratiform | 0.69 | 0.60 | 0.64 | 432 | 0.023 | 0.025 |
| low_stratus_fog | 0.86 | 0.88 | 0.87 | 1307 | 0.071 | 0.070 |
| cirrus | 0.76 | 0.80 | 0.78 | 1815 | 0.098 | 0.102 |
| convective_storm | 0.70 | 0.53 | 0.60 | 913 | 0.050 | 0.046 |
| tropical_cyclone | 0.23 | 0.50 | 0.32 | 12 | 0.0007 | 0.0032 |
| frontal_band | 0.57 | **0.01** | 0.02 | 358 | 0.019 | 0.018 |
| other | 0.84 | 0.87 | 0.86 | 10378 | 0.563 | 0.557 |

**Storm-centred test tiles** (339 tiles within 80 km of a best-track centre): tropical-cyclone recall is 0.60
and precision 0.99. Of the misses, 91 go to `other` and 47 to `convective_storm`. They are mostly weak,
sheared or landfalling storms.

How to read these:
* The model is already well calibrated after fine-tuning: the fitted temperature is T ≈ 0.98, and test ECE
  is 0.008 both before and after temperature scaling. Class-wise mean probabilities match class frequencies
  within about 10 %, except `tropical_cyclone`. That class was deliberately oversampled (storm-centred
  tiles), so its probability on a *random* tile is about 4–5× too high. In exchange, a query at a storm's
  location gets a useful TC probability. `scripts/train.py --prior-correction` stores a logit offset that
  gives the opposite trade-off: random-tile calibration, but far lower TC recall.
* `frontal_band` is effectively not learned (recall 0.01). Its probability is still calibrated *as a base
  rate*, but the model cannot pick out fronts in a 256 km IR tile from these labels. See Limitations.
* An earlier version of the split had no gap between training and held-out blocks, so Erin and Melissa
  partly leaked into training. Removing that leak changed overall test accuracy by less than 0.5 points,
  but lowered storm-centred TC recall from 0.70 to 0.60. The figures above are the leak-free ones.
* Spot checks on real test-period imagery, from `classify --lat --lon --time`:
  - Hurricane Melissa near Jamaica, 2025-10-28 12Z: `tropical_cyclone` 0.996. The same tile as PNG through
    the C13-only model: 0.99.
  - Peruvian stratocumulus deck (20°S 82°W): `low_stratus_fog` 0.92.
  - Northern Mexico desert, 2025-11-01 18Z: `clear` 0.99.
  - Weak TS Sonia via GOES-18 (not trained on): `convective_storm` 0.72, `tropical_cyclone` 0.06.

## Usage

Everything runs in Docker (the host only needs Docker):

```bash
docker build -t satwx .
```

```bash
docker run --rm satwx --lat 17.5 --lon -78.1 --time 2025-10-28T12:00
```

```bash
docker run --rm -v "$PWD:/work" satwx /work/tile.png --json
```

Inputs:
* `--lat --lon --time`: finds the nearest full-disk scan on `noaa-goes19` (or `noaa-goes18` west of
  106°W). It reads only the bytes for the 256 km tile around the point with HTTP range requests (about 8 MB,
  10–20 s), and classifies it. All four bands come from the same scan. Points off the disk or too close
  to its edge are refused. Points beyond the training range (satellite zenith > 65°) get a warning in the
  output. `--save-tile tile.npy|tile.png` also keeps the tile.
* `tile.npy`: a `(4, 64, 64)` array of C07/C08/C13/C15 brightness temperatures in K (4 km pixels).
* `tile.png`: 8-bit greyscale 10.3 µm IR, white = cold, linear from 0 → 330 K to 255 → 170 K, covering
  about 256 km. It is resized to 64×64 and classified with the C13-only model. The PNG must follow this
  mapping: colour-enhanced web imagery (e.g. Worldview/GIBS colour tables) will not give meaningful results.

Rebuild the dataset and models from scratch (about 1.5 h of downloads, roughly 25 GB fetched and deleted as it goes, plus about
1 h of training per model on 8 CPU cores; the tiles take 1.4 GB):

```bash
docker build -t satwx . && docker run --rm --entrypoint python -v "$PWD:/app" -w /app satwx scripts/build_dataset.py --every-days 4 --workers 6
```

```bash
docker run --rm --entrypoint python -v "$PWD:/app" -w /app satwx scripts/build_dataset.py --tc-only --start 2025-06-01 --end 2025-11-30 --every-days 1 --workers 6
```

```bash
docker run --rm --entrypoint python -v "$PWD:/app" -w /app satwx scripts/train.py --out models/satwx_ir4.pt
```

```bash
docker run --rm --entrypoint python -v "$PWD:/app" -w /app satwx scripts/train.py --out models/satwx_ir1.pt --channels C13
```

Tests: `docker run --rm --entrypoint python satwx -m pytest -q tests`

Layout: `satwx/goes.py` (S3 access, fixed-grid projection, readers), `satwx/scene.py` (scene loading, tiling),
`satwx/ancillary.py` (IBTrACS, WPC fronts), `satwx/labels.py` (rules), `satwx/model.py` (network, calibration),
`satwx/cli.py`, `scripts/fetch_goes.py` (stand-alone fetcher), `scripts/build_dataset.py`, `scripts/train.py`.

## Limitations

* **Accuracy is agreement with rule-based labels, not with human analysts.** The labels come from NOAA L2
  retrievals, GLM, best tracks and WPC fronts, and inherit their errors. Examples: ACHA heights are poor under
  thin cirrus, and the cloud mask struggles over snow and ice and at the terminator. A high score means the
  CNN reproduces those products from IR imagery alone. It does not mean the class names are meteorologically
  perfect. No human-labelled evaluation set was available.
* **Fog vs. low stratus cannot be separated from satellite alone**, because cloud base is not observed. Hence the
  merged `low_stratus_fog` class, which also contains marine stratocumulus decks.
* **`frontal_band` is the weakest class.** WPC fronts are surface analyses, not cloud features: a front with no
  cloud band still gets labelled when the tile is cloudy, and cloud bands away from analysed fronts do not. WPC
  fronts also only cover North America and adjacent oceans.
* **`other` is a large catch-all** (about 55 % of tiles): multi-layer and mixed scenes, decaying convection,
  and convection with only a little lightning. Expect substantial probability mass on `other` for complex scenes.
* **One satellite, one year, IR only.** Trained on GOES-19 (June 2025 to May 2026) at satellite zenith
  < 65°. GOES-18 has the same instrument and works through the same code but was not evaluated. Himawari,
  Meteosat and polar regions are not supported. Visible-band texture (useful for cumulus) is not used, so the
  model behaves the same day and night.
* **Tropical cyclone labels depend only on distance to the best-track centre** (≤ 200 km), not on structure,
  so weak or sheared storms count as TCs and the outer bands of large ones do not. Test TC tiles come from a
  few storms, so per-class numbers for TC have wide uncertainty.
* **Tile scale is fixed** at about 256 km. Features much smaller (isolated cells) or larger (synoptic fronts)
  than a tile are only partly seen.
* **Calibration** comes from temperature scaling on validation scenes from the same year and satellite. It
  will degrade under distribution shift (another season, satellite, or a PNG with a different enhancement).
* **GLM flashes are placed with the surface ellipsoid** rather than GLM's lightning ellipsoid, which shifts
  them about 10–30 km toward the limb. This only matters for flashes near tile edges, but it adds some noise
  to the `convective_storm` / `other` boundary.
* The `--lat/--lon/--time` path makes one query per scan and band (about 8 MB). That is fine for ad-hoc use;
  for bulk inference, download whole scenes and tile them as `scripts/build_dataset.py` does.
