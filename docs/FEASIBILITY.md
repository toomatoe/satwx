# Feasibility: classifying weather patterns from satellite imagery with class probabilities

**Verdict: feasible, with an important caveat about labels.** Free, global, near-real-time geostationary
imagery is easy to get, and small pretrained CNNs run fine on CPU. What is missing is a public, *labelled*
dataset covering the requested taxonomy (clear / cumulus / stratiform / fog / convective / tropical cyclone /
frontal band). The workable route is **weak supervision**: generate labels from independent physical products
(NOAA L2 cloud retrievals, GLM lightning, IBTrACS best tracks, WPC analysed fronts) and train an image model
on raw IR radiances to reproduce them. This repo does that. Its held-out accuracy therefore measures agreement
with those rule-based labels, not with human meteorologists (see README "Limitations").

Checked October 2026.

## 1. Free satellite data

| Source | Access | Verdict |
|---|---|---|
| **GOES-16/18/19 ABI + GLM, NOAA Open Data on AWS** (`s3://noaa-goes19`, `noaa-goes18`, `noaa-goes16`) | Anonymous HTTPS/S3, no account. Full disk every 10 min, CONUS every 5 min; L1b radiances, L2 CMI per band, and L2 products (clear-sky mask ACM, cloud-top phase ACTP, height ACHA, …) plus GLM lightning every 20 s | **Chosen.** Verified by listing and downloading: a 2 km full-disk band is about 24 MB and downloads in a few seconds. GOES-19 has been GOES-East (75.2°W) since April 2025; GOES-18 is West. |
| Himawari-9 AHI on AWS (`s3://noaa-himawari9`) | Anonymous; `AHI-L1b-FLDK`, `AHI-L2-FLDK-Clouds` | Good for the W Pacific (typhoons). Different file format (HSD segments), so not used here. |
| EUMETSAT (MSG SEVIRI, MTG FCI) Data Store | Free, but needs registration and an API key | Fine for Europe and Africa. Not used, because it needs credentials. |
| NASA Worldview / GIBS | Anonymous WMTS/WMS (verified HTTP 200) | Rendered *colour-mapped* PNG/JPEG tiles, not physical values. Good for viewing, weak for training. `classify image.png` accepts a greyscale IR PNG with a documented BT mapping instead. |

## 2. Labelled datasets

| Dataset | What it labels | Fit to requested classes |
|---|---|---|
| Rasp et al. 2020, *Sugar, Flower, Fish or Gravel* (Zooniverse crowd labels, MODIS Aqua/Terra Worldview images; [github.com/raspstephan/sugar-flower-fish-or-gravel](https://github.com/raspstephan/sugar-flower-fish-or-gravel)) | 4 patterns of *trade-wind shallow cumulus organisation* (bounding boxes) | Poor. Covers only shallow-convection morphology over the tropical Atlantic, not storms, TCs, fog or fronts. |
| Kaggle "Understanding Clouds from Satellite Images" (2019) | Same 4 classes as segmentation masks, about 5.5k images | Same mismatch. Download also needs a Kaggle account and accepting the competition rules, so it was not used. |
| SEVIR (`s3://sevir`, anonymous; verified: ir069, ir107, vis, vil, lght) | 4 h × 384 km storm *events* over CONUS (GOES-16 IR/VIS + NEXRAD VIL + GLM), with NOAA Storm Events metadata | Good for convective storms, but there are no negative classes, and the events are pre-selected severe weather. |
| TCIR (NTU; 70k frames, 1,285 TCs, IR/WV/VIS/PMW) · HURSAT-B1 (NCEI, 1978–2015) · Digital Typhoon (Himawari, 1978–present) | Storm-centred TC imagery with best-track intensity | Excellent *positives* for tropical cyclones, with no other classes. IBTrACS (used here) gives the same positions for any scene. |
| LSCIDMR (Bai et al., IEEE TCYB 2021; Himawari-8) | About 40k patches covering ~11 weather systems (tropical/extratropical cyclone, front, westerly jet, snow, low water cloud, high ice cloud, …) | Closest taxonomy to the request, but the paper is not open access and no anonymous download could be confirmed. |
| CloudSat/CALIPSO-collocated ABI (e.g. DeepCTC, Gorooh et al. 2020) | Vertical cloud type (Cu, Sc, St, As, Ns, Cb, Ci…) along narrow radar tracks | Physically best "cloud type" labels, but sparse (nadir curtains) and heavy to collocate. A future upgrade. |

**Conclusion:** no single open dataset covers the requested classes. The pattern classes need either human
labelling or weak supervision from physical products. This project uses weak supervision.

## 3. Models / state of the art

* Cloud-*pattern* classification: on the Kaggle/Rasp data, CNN segmenters (U-Net/EfficientNet) reach a Dice of
  roughly 0.6–0.67, limited by label noise from the crowd. Rasp et al. report that inter-annotator agreement
  itself is modest.
* Cloud-*type* classification from ABI: DeepCTC (CloudSat-trained DNN on GOES-16) and similar models perform well on
  deep and high clouds but struggle with low clouds. Operational NOAA L2 products (ACM, ACTP, ACHA) remain the
  practical pixel-level reference.
* Foundation models: SatVision-TOA (MODIS-pretrained SwinV2, transferred to ABI chips), NASA/IBM Prithvi-WxC
  (trained on MERRA-2 reanalysis, not imagery). Neither ships a weather-regime classifier head, and both are
  heavy for CPU. An ImageNet-pretrained ResNet-18 fine-tuned on IR tiles is a reasonable small baseline. It is
  what this repo uses.
* TC detection from IR is close to solved (high recall near the storm core). TC *intensity* estimation (ADT,
  DeepMicroNet, models on TCIR) has RMSE of about 10–13 kt.

## 4. What was built (see README)

GOES-19 full disk → 256 km tiles (4 IR bands, 64×64) → weak labels from ACM / ACTP / ACHA / GLM / IBTrACS / WPC
fronts → ResNet-18 fine-tune → temperature scaling on validation scenes → evaluation on held-out scenes at
least 4 days away from any training scene.

## Sources

- NOAA GOES on AWS: https://registry.opendata.aws/noaa-goes/ ; bucket listing https://noaa-goes19.s3.amazonaws.com/
- Himawari on AWS: https://registry.opendata.aws/noaa-himawari/
- EUMETSAT Data Store: https://data.eumetsat.int/
- NASA GIBS: https://nasa-gibs.github.io/gibs-api-docs/
- Rasp et al. 2020, BAMS, "Combining crowdsourcing and deep learning to explore the mesoscale organization of shallow convection"; https://github.com/raspstephan/sugar-flower-fish-or-gravel
- Kaggle Understanding Clouds: https://www.kaggle.com/c/understanding_cloud_organization
- SEVIR: Veillette et al. 2020 (NeurIPS); https://registry.opendata.aws/sevir/
- TCIR: https://www.csie.ntu.edu.tw/~htlin/program/TCIR/
- HURSAT: https://www.ncei.noaa.gov/products/hurricane-satellite-data
- Digital Typhoon: https://arxiv.org/abs/2311.02665
- LSCIDMR: Bai et al. 2021, IEEE Trans. Cybernetics, doi:10.1109/TCYB.2021.3080121
- DeepCTC: Gorooh et al. 2020, Remote Sensing 12(2):316
- SatVision-TOA: https://arxiv.org/abs/2411.17000
- IBTrACS v04r01: https://www.ncei.noaa.gov/products/international-best-track-archive
- WPC coded surface bulletins via IEM AFOS archive: https://mesonet.agron.iastate.edu/wx/afos/
