# 🩺 Polyp Detection in GI Endoscopy

This repository provides the **benchmark for polyp object detection** in gastrointestinal (GI) endoscopy images.  
Unlike existing works that primarily focus on **polyp segmentation**, this repo collates multiple datasets into a unified format and benchmarks **state-of-the-art (SOTA) object detection models** such as YOLO and RT-DETR.

The benchmark aims to serve as a reference point for future research in automated GI endoscopy analysis.
All **training logs, evaluation metrics, and visualizations** are tracked via [Comet ML](https://www.comet.com/lakshminarayanan-m/benchmarking), and all relevant code is documented in this repository.

## Dataset Description

The dataset information and preprocessing codes are provided in a separate repository.  
The models were tested in two experimental settings:  

- **A. INTERDATASET (173,450 images)** – Training, validation, and testing across 5 large datasets combined.  
- **B. INTRADATASET (1,188 images)** – Benchmark testing on unseen datasets to evaluate generalizability.  

---

### A. INTERDATASET

The consolidated dataset contains **173,450 images** derived from **5 public datasets**:

| Dataset                  | Total Images | Polyp Images | Multiple Polyp Images | Non-Polyp Images |
|--------------------------|-------------:|-------------:|----------------------:|-----------------:|
| **Kvasir-SEG**           | 1,000        | 1,000        | 48                    | 0                |
| **PolypGen**             | 1,473        | 1,347        | 123                   | 126              |
| **LD-PolypVideo**        | 40,186       | 33,875       | 2,360                 | 6,311            |
| **KUMC (PolypSet)**      | 37,899       | 35,996       | 1                     | 1,903            |
| **Real-COLON (Balanced)**| 92,892       | 46,061       | 3                     | 46,831           |

These datasets were merged into a single benchmark and split into:  
- **Train:** 121,414 images  
- **Validation:** 34,690 images  
- **Test:** 17,346 images  

---

### B. INTRADATASET

To assess **cross-dataset generalization**, we tested the trained models on **benchmark datasets not seen during training**:

| Benchmark Dataset        | Total Images | Polyp Images | Multiple Polyp Images | Non-Polyp Images |
|--------------------------|-------------:|-------------:|----------------------:|-----------------:|
| **CVC-ClinicDB**         | 612          | 612          | 30                    | 0                |
| **CVC-ColonDB**          | 380          | 380          | –                     | 0                |
| **ETIS-LaribDB**         | 196          | 196          | 6                     | 0                |

---
## Models & Training

- Implemented **YOLO (v11L)**, **YOLO (v12L)** and **RT-DETR-L** for benchmarking.  
- Training performed on the **INTERDATASET (173,450 images)**.  
- Evaluation on both **inter-dataset splits** and **intra-benchmark datasets**.  
- All metrics tracked with **Comet ML** (training loss, F1-score, PR curves, precision, recall, confusion matrices).  

---

## Metrics & Visualizations

- **Training curves**: F1, precision, recall, PR-curves.  
- **Evaluation metrics**: AP@0.5, AP@0.75, confusion matrices, Model Fitness. 
- **Cross-dataset generalization**: Tested on unseen datasets.  
- Results available under:
  - `runs/detect/benchmark/` → INTERDATASET experiments  
  - `INTERDATASET/` and `INTRADATASET/` → Evaluation outputs
  - The metrics were recorded onto: https://www.comet.com/lakshminarayanan-m/benchmarking

---

## Results

A. INTERDATASET 
| Model        | Precision | Recall | F1 Score | AP@0.5  | AP@0.75 | Model Fitness |
|--------------|---------:|--------:|---------:|-------:|-------:|-------:|
| **RTDETR-L** | 0.9450  |    0.8493    |    0.8946   |  0.9096  | 0.8303 | 0.7286| 
| **YOLOv11-L**  |  0.9945     |   0.7121     |   0.8299    |   0.8540  | 0.8289 | 0.7301 |
| **YOLOv12-L**   |  0.9928   |   0.7293    | 0.8409  |  0.8618  | 0.8348 | 0.7365 |

B. INTRADATASET 

a) ETIS
| Model        | Precision | Recall | F1 Score | AP@0.5  | AP@0.75 | Model Fitness |
|--------------|---------:|--------:|---------:|-------:|-------:|-------:|
| **RTDETR-L** | 0.9097  |  0.6298    |    0.7443   | 0.7830 |0.7034 | 0.6151| 
| **YOLOv11-L**  |  0.9891     |   0.4375     |   0.6067   |   0.7154  | 0.6953 | 0.6242 |
| **YOLOv12-L**   |  1.0000  |  0.3606   | 0.5300  |  0.6803  |0.6730 | 0.5920 |

b) CVC-ClinicDB
| Model        | Precision | Recall | F1 Score | AP@0.5  | AP@0.75 | Model Fitness |
|--------------|---------:|--------:|---------:|-------:|-------:|-------:|
| **RTDETR-L** | 0.9016  | 0.6517  | 0.7775 | 0.8142 |0.6464| 0.5844| 
| **YOLOv11-L**  |  0.9634 |   0.4375     |   0.6067   |   0.7154  | 0.6953 | 0.6242 |
| **YOLOv12-L**   | 0.9624 | 0.6331  |0.7638|0.8053| 0.6814 |0.6023|

c) CVC-ColonDB
| Model        | Precision | Recall | F1 Score | AP@0.5  | AP@0.75 | Model Fitness |
|--------------|---------:|--------:|---------:|-------:|-------:|-------:|
| **RTDETR-L** | 0.9515 | 0.7226 | 0.8214 | 0.8502 |0.7426|0.6335| 
| **YOLOv11-L**  | 0.9818 | 0.5684 |0.7200 | 0.7801 |0.7501 | 0.6582 |
| **YOLOv12-L**   | 0.9824 | 0.5868  | 0.7348 | 0.7893 | 0.7638 | 0.6644 |

## Installation
```bash
git clone https://github.com/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

##  Caveat  
This work was done solely for **documentation of personal research** and for **keeping track of dataset preparation steps**.  
It is **not intended to be fully reproducible** or directly used as a standardized dataset.  

For any clarifications, discussions, or further details, feel free to reach out.  

---

## Contact  
- **Name**: M. Lakshminarayanan  
- **Email**: lakshminarayanan.m678@gmail.com

---

## References  
- **Kvasir-SEG Dataset** – https://datasets.simula.no/kvasir-seg/
- **CVC-ClinicDB** – https://www.dropbox.com/scl/fi/ky766dwcxt9meq3aklkip/CVC-ClinicDB.rar?dl=0&e=1&file_subpath=%2FCVC-ClinicDB&rlkey=61xclnrraadf1niqdvldlds93
- **CVC-ColonDB** - https://www.kaggle.com/datasets/longvil/cvc-colondb
- **ETIS-Larib Polyp DB** – https://service.tib.eu/ldmservice/dataset/etis-larib-polyp-db  
- **KUMC PolypSet** – https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FCBUOR
- **PolypGen Dataset** – [https://polypgen.github.io/](https://polypgen.github.io/)  

---
