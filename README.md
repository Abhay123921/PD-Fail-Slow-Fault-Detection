# PD-Fail-Slow-Fault-Detection# Workload-Aware Detection and Localization of Fail-Slow Faults in PD-Disaggregated LLM Inference Systems

**Course:** CS5903 – Distributed AI Training  
**Institution:** Indian Institute of Technology Hyderabad  
**Project ID:** 25  
**Group:** 21

## 1. Overview

Disaggregated Large Language Model (LLM) inference separates **prefill** and **decode** computation into different worker pools. This architecture can improve resource utilization, but it also introduces additional communication and synchronization stages.

A slowdown may originate from compute degradation, communication bottlenecks, or synchronization delays. Identifying the responsible component is difficult because workload variation and resource contention can also increase latency.

This project develops a simulator-based framework for **workload-aware detection and localization of fail-slow faults** in prefill-decode (PD) disaggregated LLM inference systems.

The framework models the inference pipeline, collects stage-level telemetry, learns healthy workload-conditioned references, detects persistent anomalies, ranks potential fault locations, and supports abstention when the available evidence is insufficient.

> **Scope:** This is a software simulation study. It does not evaluate a deployment on physical GPUs or a production LLM serving system.

## 2. System Architecture

The simulated system contains:

- **Prefill workers:** P1 and P2
- **Decode workers:** D1 and D2
- **Communication links:** L1 and L2

The request execution pipeline is divided into six measurable stages:

1. Prefill queue wait
2. Prefill execution
3. KV-cache transfer
4. Handoff and synchronization wait
5. Decode admission wait
6. Decode execution

The framework uses stage-level telemetry to distinguish where latency increases occur.

## 3. Detection and Localization Pipeline

The implemented pipeline consists of the following stages:

1. **Healthy reference modelling:** Learns expected stage behaviour under different workload conditions.
2. **Residual computation:** Compares observed stage metrics against their healthy reference values.
3. **Anomaly detection:** Identifies stage-level deviations using calibrated thresholds.
4. **Temporal persistence:** Filters transient anomalies using persistence-based detection.
5. **Evidence assessment:** Evaluates the available evidence and its quality.
6. **Fault attribution:** Ranks potential fault locations using anomaly evidence.
7. **Diagnosis:** Produces a diagnosis or abstains when the evidence is insufficient.

For an observation \(y_{i,m}\) and its expected reference \(\hat{y}_{i,m}\), the residual is:

\[
r_{i,m}=y_{i,m}-\hat{y}_{i,m}
\]

The evidence-quality score is a heuristic score, not a calibrated probability. Similarly, component attribution represents evidence-based ranking rather than causal proof.

### Heterogeneous communication links

The project additionally evaluates heterogeneous communication links:

- L1: PCIe, configured bandwidth 20 MB/ms
- L2: NVLink, configured bandwidth 80 MB/ms

A global transfer threshold can be inadequate when healthy latency distributions differ across links. An optional link-conditioned transfer-threshold mode therefore calibrates separate healthy thresholds for known link configurations.

## 4. Fault Scenarios

The integrated experiments evaluate three controlled fault categories:

| Fault category | Injected location |
|---|---|
| Compute degradation | Prefill worker P1 |
| Communication degradation | Communication link L1 |
| Synchronization delay | D2 handoff |

Additional balanced two-link experiments evaluate communication degradation on L1 and L2 separately.

## 5. Experimental Results

The following are selected results from the completed simulator experiments.

### 5.1 Healthy workload characterization

Five workload profiles were evaluated using two seeds and 1,000 requests per profile and seed.

| Workload profile | Mean E2E latency (ms) | P95 E2E latency (ms) |
|---|---:|---:|
| Stable | 17.20 | 17.20 |
| High arrival | 21.30 | 33.87 |
| Long prompt | 70.74 | 121.03 |
| Long output | 98.98 | 194.99 |
| Mixed | 141.36 | 345.66 |

The results demonstrate that workload variation alone can substantially increase end-to-end latency. This motivates workload-conditioned healthy references.

### 5.2 Integrated fault detection

At 50% injected severity, the integrated detection pipeline localized all three evaluated fault categories.

| Fault category | Target-stage recall |
|---|---:|
| Compute degradation | 99.40% |
| Communication degradation | 99.80% |
| Synchronization delay | 99.60% |

Precision was 1.0 for the evaluated target fault cases.

These results apply to the controlled scenarios and configurations tested; they should not be interpreted as general production-system performance.

### 5.3 Persistence ablation

The persistence experiments compared count-based, time-aware, and density-based strategies using workload-conditioned and global healthy references.

Under the evaluated healthy communication holdout:

- Workload-conditioned ordinary density persistence produced 11 false positives per 5,000 observations.
- Workload-conditioned strict-density persistence produced 3 false positives per 5,000 observations.
- Global strict-density persistence produced 1 false positive per 5,000 observations.

Strict-density persistence reduced false positives compared with ordinary density persistence in the evaluated holdout. The selected persistence strategy was workload-conditioned healthy references combined with strict-density persistence.

### 5.4 Heterogeneous two-link localization

The global transfer threshold detected the tested L1 fault but missed the tested L2 fault because the healthy latency distributions differed between the two link configurations.

With link-conditioned transfer thresholds:

| Metric | Global threshold | Link-conditioned threshold |
|---|---:|---:|
| Healthy false-positive rate | 0% | 0% |
| L1 fault persistent detection | 99.6% | 99.6% |
| L2 fault persistent detection | 0% | 99.6% |
| Detection across both tested fault scenarios | 49.8% | 99.6% |

The improvement is **49.8 percentage points** across the two tested link-fault scenarios.

This demonstrates the importance of accounting for heterogeneous communication configurations when calibrating healthy transfer behaviour.

## 6. Repository Structure

```text
PD-Fail-Slow-Fault-Detection/
│
├── configs/
│   ├── detection.yaml
│   ├── faults.yaml
│   ├── system.yaml
│   └── workload.yaml
│
├── data/
│
├── docs/
│   ├── Project_Report.pdf
│   └── Architecture_Decision_Records.md
│
├── results/
│   └── final/
│       ├── healthy_characterization/
│       ├── integrated_fault_experiments/
│       ├── persistence_strict_density_ablation/
│       ├── balanced_two_link_localization/
│       ├── balanced_two_link_localization_l2_fault/
│       └── balanced_two_link_localization_link_conditioned/
│
├── scripts/
│   ├── run_balanced_two_link_localization.py
│   ├── run_integrated_fault_experiments.py
│   ├── run_persistence_three_way_ablation.py
│   ├── run_fault_severity_sweep.py
│   └── ...
│
├── src/
│   └── pd_failslow/
│       ├── detection/
│       ├── diagnosis/
│       ├── evaluation/
│       ├── faults/
│       ├── simulator/
│       ├── telemetry/
│       ├── utils/
│       └── workload/
│
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

`results/final/` contains selected compact experiment evidence. The original experiment directories are retained separately and are not all included in the curated evidence set.

## 7. Installation

### Requirements

- Python 3.10 or later
- pip
- Git

Clone the repository:

```bash
git clone https://github.com/Abhay123921/PD-Fail-Slow-Fault-Detection.git
cd PD-Fail-Slow-Fault-Detection
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install the project in editable mode:

```bash
python3 -m pip install -e .
```

## 8. Running Experiments

Run commands from the repository root with the virtual environment activated.

### Integrated fault experiments

```bash
python3 scripts/run_integrated_fault_experiments.py
```

### Persistence ablation

```bash
python3 scripts/run_persistence_three_way_ablation.py
```

### Balanced two-link localization

```bash
python3 scripts/run_balanced_two_link_localization.py
```

### Fault severity sweep

```bash
python3 scripts/run_fault_severity_sweep.py
```

**Experiment prerequisites:** The integrated, persistence, and balanced two-link experiments rely on previously generated healthy-characterization traces. Persistence ablation also requires integrated experiment traces and ground-truth files. The corresponding inputs must exist under the expected `results/` directories before running these scripts.

These scripts use fixed configuration and output paths. Back up existing experiment outputs before rerunning them, as generated files may be overwritten.

## 9. Running Tests

Run the test suite from the repository root:

```bash
python3 -m pytest -q
```

The test suite covers simulation, fault injection, detection, persistence, communication diagnostics, localization, evaluation, and related components.

## 10. Limitations

The current study has the following limitations:

- The system is simulator-based and has not been validated on physical GPU infrastructure.
- The simulation uses a deterministic two-pass execution model rather than a fully event-driven serving implementation.
- Fault localization results are based on a limited set of controlled fault scenarios.
- Link-conditioned calibration has been evaluated on one PCIe and one NVLink configuration.
- Unseen link configurations, additional workload distributions, and a broader range of fault severities require further evaluation.
- Evidence-quality and attribution scores are heuristic and should not be interpreted as calibrated probabilities or causal guarantees.

## 11. Future Work

Potential extensions include:

- Evaluation on real PD-disaggregated LLM serving infrastructure.
- Broader workload and fault-severity evaluation.
- Generalization to unseen communication configurations.
- More detailed modelling of network contention and KV-cache transfer.
- Online adaptation of healthy references and anomaly thresholds.
- Stronger confidence calibration and fault attribution methods.

## 12. Project Information

**Project:** Workload-Aware Detection and Localization of Fail-Slow Faults in PD-Disaggregated LLM Inference Systems

**Course:** CS5903 – Distributed AI Training

**Project ID:** 25

**Group:** 21

**Institution:** Indian Institute of Technology Hyderabad

**GitHub:** https://github.com/Abhay123921/PD-Fail-Slow-Fault-Detection
