---
license: mit
pretty_name: 5G Network Energy Consumption Dataset
configs:
- config_name: BSinfo
  description: Base Station Basic Information
  data_files:
  - split: test
    path: BSinfo.csv
- config_name: CLstat
  description: Cell-Level Statistics
  data_files:
  - split: test
    path: CLstat.csv
- config_name: ECstat
  description: Energy Consumption Statistics
  data_files:
  - split: test
    path: ECstat.csv
extra_gated_fields:
  First Name: text
  Last Name: text
  Affiliation: text
  Country: country
  Job title:
    type: select
    options:
      - Student
      - Research Graduate
      - Researcher
      - Engineer
      - Reporter
      - Other
    geo: ip_location
language:
- en
tags:
- 5G
- wireless
- energy
---

# 5G Network Energy Consumption Dataset

This dataset provides normalized real-world measurements of energy consumption and operational data from a large-scale 5G network deployment. 
It includes eight days of measurements collected from **more than 1,000 RRU/AAUs**, covering **12 different hardware products**.

The dataset is intended to support research in energy-efficient mobile networks, network optimization, and data-driven modeling of 5G systems.

---

## 📂 Dataset Structure

The dataset is organized into three main CSV files:

### 1. **BSinfo.csv** – Base Station Basic Information

Contains static information about the deployed base stations and their cells.

**Fields:**

* `BS`: Identifier of the base station
* `CellName`: Cell identifier (multiple cells can be configured per BS, named `CellX`)
* `RUType`: Radio unit product name
* `Mode`: Transmission mode
* `Bandwidth`: Normalized cell bandwidth
* `Frequency`: Normalized cell frequency
* `Antennas`: Number of antennas
* `TXpower`: Maximum transmit power of the cell

---

### 2. **CLstat.csv** – Cell-Level Statistics

Contains hourly counters describing the operational status of each cell.

**Fields:**

* `Time`: Timestamp of the measurement
* `BS`: Identifier of the base station
* `CellName`: Cell identifier
* `Load`: Load of the cell (share of used resources)
* `EnergySavingMode`: Intensity of activation of energy-saving mechanisms

---

### 3. **ECstat.csv** – Energy Consumption

Contains hourly measurements of energy consumption per base station.

**Fields:**

* `Time`: Timestamp of the measurement
* `BS`: Identifier of the base station
* `Energy`: Energy consumption of the base station

---

## 📊 Data Summary

* **Measurement period:** 8 days
* **Number of RRUs/AAUs:** > 1,000
* **Hardware diversity:** 12 different product types
* **Granularity:** Hour-level measurements for load, counters, and energy consumption

---

## 🧪 Applications

This dataset can be used for:

* Analysis of energy consumption patterns in 5G networks
* Evaluation of energy-saving methods and load-dependent consumption
* Development of predictive models for network optimization
* Benchmarking of AI/ML approaches for green networking

---
