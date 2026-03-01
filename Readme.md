# Escaping the Echo  
*A CNN-Based Approach for Echo-Aware Friend Recommendation in Social Networks*

## Overview

This project implements a multimodal friend recommendation framework for social networks.  
User representations are learned from:

- **Textual content** using a Siamese Convolutional Neural Network (CNN)
- **Interaction structure** using Node2Vec graph embeddings

Both modalities are combined into a joint embedding space to enable similarity-based retrieval and echo-aware re-ranking.

All experiments were executed with a fixed random seed (`SEED = 42`) to promote reproducibility; however, full determinism cannot be guaranteed.

---

## Environment Setup

This repository includes an `environment.yml` file for reproducible installation.

### 1. Create the Conda Environment

Open a terminal in the project directory and run:

```bash
conda env create -f environment.yml
```
This command creates a new Conda environment with the specified Python version and required dependencies.

### 2. Activate the Environment

```bash
conda activate escaping_echo
```
### 3. Running the project

After activating the environment, execute the juypter notebooks.
