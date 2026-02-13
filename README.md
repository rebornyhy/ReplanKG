# RePlanKG: Report-Guided Symbolic Planning for Knowledge Graph Question Answering

[中文版](./README_zh.md)

RePlanKG is a report-guided plan–execute–reflect framework for Knowledge Graph Question Answering (KGQA). It enables large language models (LLMs) to generate executable symbolic retrieval plans, execute them over knowledge graphs, and iteratively refine reasoning using structured execution feedback.

This repository provides the official inference implementation of RePlanKG.

---

## Overview

Knowledge Graph Question Answering (KGQA) requires structured multi-hop reasoning over large knowledge graphs. Existing paradigms suffer from key limitations:

### Step-by-step KGQA
- Poor efficiency due to uncontrolled multi-step generation  
- Error accumulation  
- Ineffective global reasoning  

### Retrieval-then-answer KGQA
- Low robustness  
- Low-quality symbolic queries (e.g., SPARQL)  
- No execution feedback when retrieval fails  

RePlanKG addresses these challenges through a report-guided **plan–execute–reflect loop**, enabling:

- Explicit symbolic planning  
- Structured execution feedback  
- Iterative refinement  
- Improved reasoning robustness and efficiency  

---

## Framework

### 1️⃣ RePlanKG Architecture

<!-- Replace with your actual image path -->
![RePlanKG Framework](./figs/rpg_framework.png)

---

### 2️⃣ Paradigm Comparison

<!-- Replace with your actual image path -->
![Paradigm Comparison](./figs/vs_pre_work.png)

---

### 3️⃣ Case Study Example

<!-- Replace with your actual image path -->
![Case Study](./figs/case.png)

---

## Installation

We recommend using Python 3.10.

### Step 1: Install PyTorch (CUDA 12.4 example)

```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Step 2: Install other dependencies

```
pip install -r requirements.txt
```

---

## Download Model & Data

### 1️⃣ Download RePlanKG LLM Weights

Model is hosted on **ModelScope**:

rebornyhy/replankg_13b

After downloading, place the model folder into:

```
./replankg_llm/
```

Final structure example:

```
replankg_llm/
    config.json
    tokenizer.json
    pytorch_model.bin
    ...
```

---

### 2️⃣ Download mid2name.pkl

Download from Baidu Netdisk:

Link: https://pan.baidu.com/s/11mutP53dVkK-rWG8Olfgxg  
Password: i2ga  

After downloading:

Replace the folder:

```
./mid2name/
```

with the downloaded folder:

```
mid2name/
```

Ensure that:

```
./mid2name/mid2name.pkl
```

exists.

---

### 3️⃣ Download data_kg Files

Download from Baidu Netdisk:

Link: https://pan.baidu.com/s/1uRbvdGbp7yD8PDaJgli6lA  
Password: vi3f  

After downloading:

Replace:

```
./data_kg/
```

with the downloaded data_kg folder.

---

## Running Inference

RePlanKG provides a unified inference script:

```
kgqa_infer_args.py
```

To simplify usage, we provide:

```
run_replankg_infer.sh
```

### Usage

1. Open:

```
run_replankg_infer.sh
```

2. Modify required parameters at the top of the script:

- LLM_PATH  
- MID2NAME_PATH  
- EMB_MODEL_PATH  
- DATA_PATH  
- EXP_NAME  
- OUTPUT_DIR  
- KG_CLASS (simpleques / webqsp / cwq)  

3. Run:

```
bash run_replankg_infer.sh
```

The script automatically assembles the command and executes inference.

---

## Project Structure

```
.
├── kgqa_infer_args.py        # Main inference script
├── run_replankg_infer.sh     # Inference launcher
├── replankg_llm/             # LLM weights (download separately)
├── data_kg/                  # KG data (download separately)
├── mid2name/                 # mid2name.pkl (download separately)
├── requirements.txt
└── ...
```

---

## Supported Datasets

- SimpleQuestions  
- WebQSP  
- ComplexWebQuestions (CWQ)  

Each dataset requires specific data_kg files. See run_replankg_infer.sh for dataset-specific arguments.

---

## Reproducibility Notes

- Default reasoning width: 3  
- Default reflection iterations: 1  
- Hop is automatically determined by dataset:
  - simpleques → 1-hop  
  - webqsp → 2-hop  
  - cwq → 4-hop  

---

## Citation

If you find this repository useful, please cite:

```
@article{replankg2025,
  title={RePlanKG: Report-Guided Symbolic Planning for Knowledge Graph Question Answering},
  author={Anonymous},
  year={2025}
}
```

---

## License

This project is released under the MIT License.

---

## Contact

For questions or collaboration, please open an issue or contact the authors.
