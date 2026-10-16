# Workload-Aware Detection and Localization of Fail-Slow Faults in PD-Disaggregated LLM Inference Systems

**Course:** Distributed AI Learning  
**Student:** Abhay Raj Singh  
**Roll Number:** AI26RESCH11004  
**Institution:** Indian Institute of Technology Hyderabad

## 1. Overview

Large Language Model (LLM) inference systems increasingly use **Prefill-Decode (PD) disaggregation**, where prefill and decode operations run on separate workers. This separation allows the two stages to be managed independently, but it also introduces additional components and communication paths that can affect end-to-end latency.

A key challenge is detecting **fail-slow faults**: situations where a worker or communication link continues to operate but performs more slowly than expected.

Not every increase in latency indicates a fault. Longer prompts, different batch sizes, increased request arrivals, execution configurations, and resource contention can also increase latency.

This project proposes a workload-aware approach to detecting candidate slowdowns, distinguishing workload-induced latency from possible component degradation, and localizing faults to prefill workers, decode workers, or KV-cache transfer links.

> **Current status:** Problem analysis, literature study, and system design have been completed. Simulator implementation, fault injection, detection algorithms, and quantitative evaluation are planned next steps.

## 2. Problem Statement

Conventional fixed-threshold monitoring can generate false alarms when workload characteristics change. Historical baselines can become contaminated by faults, while peer-based comparisons can fail when multiple workers experience the same degradation.

The central research question is:

**How can fail-slow faults be detected and localized in PD-disaggregated LLM inference systems without confusing normal workload variation with component-induced performance degradation?**

The project investigates whether workload-aware references, multiple sources of system evidence, and confidence-weighted historical and peer comparisons can improve detection and localization compared with simpler baselines.

## 3. System Model

The target system consists of three main component types:

- **Prefill workers:** Process input tokens and generate the Key-Value (KV) cache.
- **KV-cache transfer links:** Transfer KV-cache data from prefill workers to decode workers.
- **Decode workers:** Generate output tokens using the input context and KV cache.

A simplified request path is:

```text
Incoming Request
       |
       v
 Prefill Worker
       |
       | KV-cache transfer
       v
 Transfer Link
       |
       v
  Decode Worker
       |
       v
 Generated Response
```

Slowdown in any stage can affect request latency. The objective is to identify abnormal component behaviour while accounting for the workload and execution conditions under which the request was processed.

## 4. Project Objectives

1. Model a simplified PD-disaggregated LLM inference system.
2. Collect stage-level latency, workload characteristics, execution settings, and component-level telemetry.
3. Establish healthy reference behaviour under different workload conditions.
4. Detect candidate slowdowns using observed behaviour relative to an appropriate reference.
5. Attribute detected anomalies to workload changes, execution conditions, interference, or possible component degradation where evidence permits.
6. Localize suspected faults to prefill workers, decode workers, or KV-cache transfer links.
7. Evaluate historical, peer-based, and confidence-weighted reference strategies.
8. Measure detection quality, false alarms, detection delay, localization accuracy, and algorithmic overhead.

## 5. Proposed Methodology

The proposed pipeline contains five stages.

### Stage 1: Telemetry Collection

Collect request-level and component-level information, including:

- Request identifiers and component identifiers.
- Prompt and output characteristics.
- Batch size and request arrival conditions.
- Stage-level and per-hop latency.
- Execution configuration and relevant system state.
- Available GPU, operating-system, queue, and network measurements.

The initial implementation will use a simulator with configurable service-time and capacity models.

### Stage 2: Healthy Reference Estimation

Estimate expected behaviour from trusted healthy execution traces and, where appropriate, comparable peer components.

The reference should account for relevant workload and execution characteristics instead of relying exclusively on a single global latency threshold.

Historical and peer observations must be treated cautiously because they may themselves be affected by faults or common-mode degradation.

### Stage 3: Candidate Slowdown Detection

Compare observed latency against the expected latency for the relevant workload and execution conditions.

A robust statistical detector based on the median and Median Absolute Deviation (MAD) is an initial candidate:

\[
\operatorname{MAD}=\operatorname{median}_i
\left(|x_i-\operatorname{median}(x)|\right)
\]

A robust deviation score can then be used to identify unusually high latency relative to a reference distribution.

The exact threshold, persistence conditions, and treatment of low-confidence observations will be calibrated and evaluated during implementation. A statistical anomaly is a candidate slowdown, not proof of a fault.

### Stage 4: Cause Attribution

Investigate whether a detected slowdown is more consistent with:

- A change in workload characteristics.
- An execution configuration or resource-contention change.
- A possible prefill-worker slowdown.
- A possible decode-worker slowdown.
- A possible KV-cache transfer bottleneck.

Attribution will use available telemetry and comparisons rather than assuming that latency alone identifies the cause.

### Stage 5: Component Localization

Use stage-level timings, component observations, and relevant system evidence to identify the most likely affected component.

If the evidence is insufficient to distinguish between possible causes, the system should report an uncertain or unresolved diagnosis rather than force a potentially incorrect localization.

## 6. Confidence-Weighted References

An important direction under investigation is combining historical observations and peer observations according to their reliability and comparability.

For example, a reference weight may be represented as:

\[
w_s=R_s C_s
\]

where:

- \(R_s\) represents the estimated reliability of a source.
- \(C_s\) represents its comparability to the current workload and execution conditions.

These weights can contribute to a weighted expected-latency estimate.

However, confidence weighting is a **proposed technique to evaluate**, not an established improvement. The experiments will compare simple references, static confidence weighting, and dynamic confidence weighting to determine whether the additional complexity is justified.

## 7. Planned Implementation

The implementation is planned as a modular, simulator-driven prototype.

| Module | Responsibility |
|---|---|
| `configs/` | System, workload, and fault configurations |
| `simulator/` | PD system, workload generation, scheduling, transfer, and fault injection |
| `telemetry/` | Trace collection, feature extraction, and dataset preparation |
| `detection/` | Healthy references, anomaly detection, workload attribution, and localization |
| `evaluation/` | Experiments, detection metrics, localization metrics, and ablation studies |
| `tests/` | Unit and integration tests |
| `results/` | Experimental outputs, metrics, and plots |
| `notebooks/` | Exploratory analysis and visualization |

The simulator will initially approximate component behaviour through configurable service-time and capacity models. It is intended to provide controlled, reproducible traces for evaluating detection methods, rather than to reproduce every detail of real GPU execution.

## 8. Fault and Workload Scenarios

The planned evaluation will include:

**Healthy workload variation**
- Different prompt and output lengths.
- Different batch sizes and request arrival rates.
- Changes in execution configuration and interference.

**Injected faults**
- Slowdown in an individual prefill worker.
- Slowdown in an individual decode worker.
- Slowdown in a KV-cache transfer link.
- Gradual component degradation.

**Difficult diagnosis conditions**
- Common-mode slowdown affecting multiple peers.
- Potential contamination of historical references.
- Workload changes that increase latency without a fault.
- Cases where available evidence cannot reliably identify one component.

Fault injection will provide independent ground-truth labels for evaluating the proposed methods.

## 9. Evaluation Plan

The project will compare the proposed methods against simpler baselines.

| Method or metric | Purpose |
|---|---|
| Fixed-threshold detection | Establish a simple baseline |
| Historical baseline | Evaluate a component's past behaviour as a reference |
| Peer comparison | Evaluate comparisons with comparable workers |
| Static confidence weighting | Test fixed reliability and comparability weights |
| Dynamic confidence weighting | Test whether updated confidence scores improve results |
| Precision, recall, and F1-score | Measure detection quality |
| False-positive rate | Measure false alarms, particularly under healthy workload changes |
| Detection delay | Measure time from fault injection to detection |
| Localization accuracy | Measure whether the affected component is correctly identified |
| Algorithmic overhead | Measure detection and diagnosis time and memory requirements |

Experiments will distinguish development and evaluation data appropriately. Thresholds and parameters will be selected using development or validation data rather than tuned on final test results.

No quantitative performance claims are made at this stage because the planned experiments have not yet been completed.

## 10. Current Progress

### Completed

- Problem definition and analysis.
- Study of relevant literature and existing approaches.
- Definition of the PD-disaggregated system model.
- Identification of workload-induced versus component-induced slowdown.
- Design of the proposed detection, attribution, and localization pipeline.
- Identification of telemetry requirements and reference-estimation challenges.
- Planning of the simulator architecture and evaluation methodology.

### Planned

- Implement the simplified PD simulator and request path.
- Define the trace schema and generate healthy execution traces.
- Implement controlled fault injection with ground-truth labels.
- Implement fixed-threshold, historical, and peer-based detection baselines.
- Add workload-aware attribution and component localization.
- Evaluate static and dynamic confidence-weighted references.
- Report quantitative metrics and comparative experimental results.

## 11. Scope and Limitations

The initial project focuses on **detection and localization** in a simplified PD-disaggregated inference system.

The following are outside the initial implementation scope:

- Automatic mitigation and recovery.
- Worker rerouting and component quarantine.
- Production deployment and autoscaling.
- Full fidelity simulation of every GPU, accelerator, and communication stack.
- Application-level behavioural monitoring unless a suitable requirement and evaluation method are established.

Simulation results will need to be interpreted within the limits of the simulator. Good performance in controlled experiments does not automatically guarantee the same performance in a real deployment.

## 12. Expected Outcomes

The planned deliverables are:

1. A configurable simulator for a simplified PD-disaggregated inference system.
2. A trace-generation and telemetry pipeline.
3. Controlled healthy and faulty execution scenarios.
4. Baseline slowdown-detection methods.
5. Workload-aware cause attribution and component localization.
6. Comparative evaluation of historical, peer-based, and confidence-weighted references.
7. A reproducible evaluation package containing configurations, code, metrics, and plots.

These are expected outcomes, not completed experimental results.

## 13. Conclusion

This project investigates how fail-slow faults can be detected and localized in PD-disaggregated LLM inference systems despite changing workloads and execution conditions.

Its central principle is that **increased latency should be investigated relative to expected behaviour, not automatically classified as a fault**. By combining workload-aware references, component-level evidence, anomaly detection, cause attribution, and localization, the project aims to build a reproducible framework for evaluating this problem.

The effectiveness of the proposed approach will ultimately be determined by experiments comparing it with simpler detection and reference-estimation methods.

---

**Repository:** [PD-Fail-Slow-Fault-Detection](https://github.com/Abhay123921/PD-Fail-Slow-Fault-Detection)

**Keywords:** LLM Inference, Distributed Systems, Prefill-Decode Disaggregation, Fail-Slow Faults, Anomaly Detection, Workload-Aware Monitoring, KV Cache, Fault Localization, System Telemetry.
