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

#### **Database organisation**

Unlike existing visual place recognition datasets, where images are organised in a way that's not (so humanly) explorable. Images in GSV-Cities are names as follows:

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

Each datadrame contains the metadata of the its corresponding city. This will help query the dataset almost instantly using Pandas. For example, we show 5 rows from London.csv:

| place_id | year | month | northdeg | city_id |     lat |        lon | panoid                 |
| -------: | ---: | ----: | -------: | :------ | ------: | ---------: | :--------------------- |
|      130 | 2018 |     4 |       15 | London  | 51.4861 | -0.0895151 | 6jFjb3wGyCkcBfq4k559ag |
|     6793 | 2016 |     7 |        2 | London  | 51.5187 |  -0.160767 | Ff3OtsS4ihGSPdPjtlpEUA |
|     9292 | 2018 |     1 |      289 | London  |  51.531 |   -0.12702 | 0t-xcCsazIGAjdNC96IF0w |
|     7660 | 2015 |     6 |       97 | London  | 51.5233 |  -0.158693 | zFbmpj8jt8natu7IPYrh_w |
|     8713 | 2008 |     9 |      348 | London  | 51.5281 |  -0.127114 | W3KMPec54NBqLMzmZmGv-Q |

 And If we want only places that are depicted by at least 8 images each, we can simply filter the dataset using pandas as follows:

```
df = pd.read_csv('London.csv')
df = df[df.groupby('place_id')['place_id'].transform('size') >= 8]
```

Notice that given a Dataframe row, we can directly read its corresponding image (the first row of the above example corresponds to the image named `./Images/London/London_0000130_2018_04_015_51.4861_-0.0895151_6jFjb3wGyCkcBfq4k559ag.JPG`)

We can, for example, query the dataset with *only places that are in the northern hemisphere, taken between 2012 and 2016 during the month of July, each depicted by at least 16 images*.

...to be continued
