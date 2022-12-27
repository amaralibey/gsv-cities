# GSV-CITIES

Official repo for Neurocomputing 2022 paper: **GSV-Cities: Toward Appropriate Supervised Visual Place Recognition**

---

**Summary:**

We collect a large-scale dataset for visual place recognition, with highly accurate ground truth. We call it **GSV-Cities** (GSV refers to Google Street View). We also propose, a fully convolutional aggregation technique (called **Conv-AP**) that outperforms NetVLAD and most existing SotA techniques. We consider representation learning for place recognition as a three components pipeline as follows.

![1672170020629](image/README/1672170020629.png)

What can we do with GSV-Cities dataset?

* Train visual place recognition models extremely rapidly
* No offline triplet mining: GSV-Cities contains highly accurate ground truth. Batches are formed in a traightforward way, bypassing all the hassle of triplet preprocessing.
* Rapid prototyping: no need to wait days for convergence. Using GSV-Cities, the network will show convergence after one or two epochs (expect ~20 minutes of training per epoch when using half precision).
* All existing techniques benefits from training on GSV-Cities.

---

## GSV-Cities dataset overview

* GSV-Cities contains ~500,000 images representing ~67,000 different places, all spread across multiple cities around the globe.
* All places are physically distant (at least 100 meters between any pair of places).

![1672162442594](image/README/1672162442594.png)

**Database organisation**

Contrary to existing visual place recognition datasets, where images are organised in a way that's not (so humanely) explorable. Images in GSV-Cities are names as follows:

`city_place-id_year_month_bearing_latitude_longitude_panoid.JPG`

This way of naming has the advantage of exploring the dataset using the default Image Viewer of the OS, and also, adding redondancy of the metadata in case the Dataframes get lost or corrupt.

The dataset is organised as follows:

```
├── Images
│   ├── Paris
│   │   ├── ...
│   │   ├── PRS_0000003_2015_05_584_48.79733778544615_2.231461206488333_7P0FnGV3k4Fmtw66b8_-Gg.JPG
│   │   ├── PRS_0000003_2018_05_406_48.79731397404108_2.231417994064803_R2vU9sk2livhkYbhy8SFfA.JPG
│   │   ├── PRS_0000003_2019_07_411_48.79731121699659_2.231424930041198_bu4vOZzw3_iU5QxKiQciJA
│   │   ├── ...
│   ├── Boston
│   │   ├── ...
│   │   ├── Boston_0006385_2015_06_121_42.37599246498178_-71.06902130162344_2MyXGeslIiua6cMcDQx9Vg.JPG
│   │   ├── Boston_0006385_2018_09_117_42.37602467319898_-71.0689666533628_NWx_VsRKGwOQnvV8Gllyog.JPG
│   │   ├── ...
│   ├── Quebec
│   │   ├── ...
│   ├── ...
└── Dataframes
    ├── Paris.csv
    ├── London.csv
    ├── Quebec.csv
    ├── ...

```
