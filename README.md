# DNSLLM Artifact Repository

This repository contains the artifact for the DNSLLM methodology, which integrates Dual-Graph RAG construction, DRFA, and LLM-based inference for DNS anomaly detection. The structure and content of this repository are designed to ensure reproducibility and clarity.

## Directory Structure

```
artifact/
├── data/                       # Input datasets and auxiliary resources
│   ├── datasets/               # DNS datasets used in experiments
│   │   ├── dataset1_sample.jsonl
│   │   ├── dataset2_sample.jsonl
│   │   ├── dataset3_sample.jsonl
│   │   ├── dataset4_sample.jsonl
│   │   ├── normal_sample.jsonl
│   │   ├── dataset_sample.jsonl
│   │   ├── monitor_domain_info.csv
│   │   ├── policy_country.csv
│   │   └── policy.csv
│   ├── GeoLite2-ASN_20250702/        # GeoLite2 ASN database
│   ├── GeoLite2-City_20250702/       # GeoLite2 City database
│   └── GeoLite2-Country_20250702/    # GeoLite2 Country database
│
├── graph_construction/         # Graph construction notebooks
│   ├── entity_graph_construction.ipynb
│   └── resolution_graph_construction.ipynb
│
├── dual_graph_rag/             # Dual-Graph RAG pipeline
│   ├── build_answer.ipynb
│   ├── entity_graph_rag.ipynb
│   ├── extract_paths_to_jsonl.ipynb
│   ├── resolution_graph_rag.ipynb
│   ├── entity_rag.py
│   └── resolution_rag.py
│
├── drfa/                       # DNS Record Field Augmentation (DRFA)
│   └── DRFA.ipynb
│
├── outputs/                    # Generated intermediate and final outputs
│   ├── answer_gen.jsonl
│   ├── dataset_sample_with_paths.jsonl
│   ├── drfa_llm_outputs.jsonl
│   ├── entity_graph.gpickle
│   ├── ip_pool.json
│   └── resolution_graph.gpickle
│
└── README.md
```

## Datasets

All datasets are stored under `data/datasets/` in **JSONL** format (one DNS record per line).  
For artifact reproducibility and efficiency, we provide **randomly sampled subsets**.

### Dataset Summary

| Dataset File | Description | Label | #Records |
|-------------|-------------|-------|---------:|
| dataset1_sample.jsonl | Records extracted by cross-referencing probe data with threat intelligence feeds that flag suspicious IPs. | anomalous | 1000 |
| dataset2_sample.jsonl | Records extracted by cross-referencing probe data with OONI-flagged measurements and verified via web crawling. | anomalous | 1000 |
| dataset3_sample.jsonl | Records identified via web crawling by abnormal HTTP status codes and invalid page content. | anomalous | 1000 |
| dataset4_sample.jsonl | Records identified via web crawling as ad or jump pages based on HTML structure and redirection behavior. | anomalous | 1000 |
| normal_sample.jsonl | Benign DNS resolution records | normal | 1000 |
| dataset_sample.jsonl | Union of all sampled datasets | mixed | 5000 |

Each anomalous dataset (`dataset1`–`dataset4`) corresponds to a **distinct DNS integrity anomaly category** defined in the paper.  
Each category is randomly sampled to **1000 records** from the original dataset.

### DNS Record Format

Each line in a dataset JSONL file is a DNS record:

```json
{
  "name": "example.com",
  "timestamp": "2025-09-04T08:44:01+08:00",
  "status": "NOERROR",
  "data": {
    "resolver": "210.2.4.8:53",
    "protocol": "udp",
    "answers": [
      {"answer": "31.13.95.17", "class": "IN", "name": "example.com", "ttl": 169, "type": "A"}
    ]
  }
}
```

## DNSLLM Methodology

### 1. Data Input
- DNS historical resolution records (domain–IP mappings and resolver behavior).
- Entity attributes from open network intelligence sources (ASN, organization, country) and curated policy knowledge.

### 2. Dual-Graph Construction
- **Entity Graph**: Static relationships among domains, IPs, ASNs, organizations, countries, and policies.
- **Resolution Graph**: Temporal domain–ASN resolution dynamics with time-aware edge weights.

### 3. Dual-Graph RAG
- Retrieve multi-hop evidence paths from both graphs using beam search and temporal weighting.
- Linearize retrieved paths into structured evidence for LLM input.

### 4. DNS Record Field Augmentation (DRFA)
- Apply perturbation-guided augmentation to non-semantic DNS fields (e.g., TTL, ports, resolver IPs, answer IPs).
- Enforce semantic equivalence via constrained candidate sets.
- Use **LLM self-scoring** to select top-ranked augmented samples.

### 5. LLM Inference
- Structured prompts combining DNS records and dual-graph evidence.
- A local LLM (e.g., LLaMA3-8B) is used for anomaly classification and rationale generation.

## Execution Instructions

### Graph Construction
Run the notebooks in `graph_construction/`:
- `entity_graph_construction.ipynb`
- `resolution_graph_construction.ipynb`

Outputs:
- `outputs/entity_graph.gpickle`
- `outputs/resolution_graph.gpickle`

### Dual-Graph RAG and Path Extraction
Run the notebooks/scripts in `dual_graph_rag/`:
- `entity_graph_rag.ipynb` / `entity_rag.py`
- `resolution_graph_rag.ipynb` / `resolution_rag.py`
- `extract_paths_to_jsonl.ipynb`

Output:
- `outputs/dataset_sample_with_paths.jsonl`

### DNS Record Field Augmentation (DRFA)
Run:
- `drfa/DRFA.ipynb`

Outputs:
- `outputs/ip_pool.json`
- `outputs/drfa_llm_outputs.jsonl`

### Answer Generation
Run:
- `dual_graph_rag/build_answer.ipynb`

Output:
- `outputs/answer_gen.jsonl`

## Reproducibility Notes

- Each dataset category is sampled to **1000 DNS records** to reduce computational overhead while preserving category diversity.
- GeoLite2 ASN, City, and Country databases are included for consistent IP-to-entity mapping.
- No privileged access to authoritative DNS infrastructure is required.

## Contact

For questions regarding artifact usage or reproduction details, please contact the authors.