# Escaping the Echo

*Echo-aware friend recommendation on Reddit reply networks*

## Question

Friend recommenders tend to suggest users who are already close in the social graph or similar in what they write, which reinforces echo chambers.
This project asks: **does echo-aware re-ranking widen cross-community exposure, and what does it cost in accuracy?**

## Data

- Source: the public Hugging Face dataset [`HuggingFaceGECLM/REDDIT_comments`](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments), subreddit `changemyview`, streamed with Polars and cached locally as Parquet under `data/processed/` (git-ignored).
- Licence: the dataset card does not declare a licence. The content is user-generated Reddit text and remains subject to Reddit's terms of use. Only derived aggregates are reported here; no usernames or comment bodies are shown in the notebooks or this document.
- The first 100,000 retained comments are used, split by time (80 % train, 20 % test). A reply to a comment is a positive `(user, parent author)` interaction: 45,452 train and 10,214 test pairs, with 6,622 users in the training interaction graph.

## Approach

1. **Interaction graph and triplets** - reply pairs form a directed graph; hard negatives come from the same thread (anchor, positive, negative).
2. **Text embeddings** - a Siamese CNN (multi-kernel text encoder, triplet margin loss) embeds each user's recent comments. A pretrained sentence transformer (`all-MiniLM-L6-v2`) is compared as a text encoder in a separate notebook.
3. **Graph embeddings** - Node2Vec on the training graph.
4. **Joint space and retrieval** - graph and text vectors are combined and indexed with FAISS; a candidate pool (same-thread users plus two-hop neighbours) is ranked by a weighted graph and text similarity ("Hybrid").
5. **Communities** - Louvain on the undirected, weighted training graph.
6. **Echo-aware re-ranking** - popular candidates are scored as `popularity - 0.5 * similarity(user, candidate)` in the joint space, so very similar users are pushed down.
7. **Baselines** - random, popularity, and common neighbours.

Evaluation uses new test-period reply targets of users with embeddings, at k = 10. Accuracy: Precision, Recall and nDCG. Echo metrics: individual and community diversity and novelty (distances in the joint space) and the **cross-community exposure rate**, the share of recommendations that fall outside the target user's Louvain community.

## Results

Values come from the saved outputs of `EchoAware_FriendRecommender_Reddit_Final.ipynb` (single run, seed 42, k = 10).

| Method | Precision@10 | Recall@10 | nDCG@10 | Indiv. diversity | Community diversity | Cross-community exposure |
|---|---|---|---|---|---|---|
| Random | 0.0005 | 0.0000 | 0.0002 | 1.298 | 1.299 | 0.962 |
| Popularity | 0.0129 | 0.0512 | 0.0318 | 1.011 | 1.009 | 0.953 |
| Common neighbours | 0.0082 | 0.0274 | 0.0174 | 1.117 | 1.178 | 0.813 |
| Hybrid (graph + text) | 0.0024 | 0.0118 | 0.0075 | 0.881 | 1.108 | 0.664 |
| **Popularity, echo-aware re-ranking** | 0.0096 | 0.0410 | 0.0275 | 1.189 | 1.263 | **0.969** |

Novelty (individual / community): random 1.262 / 1.282, popularity 1.101 / 1.272, common neighbours 1.072 / 1.141, hybrid 1.018 / 1.062, echo-aware 1.428 / 1.466.
An oracle that returns the true future targets first reaches Precision@10 = 0.239, the upper bound under this evaluation.

Reading the table:

- Echo-aware re-ranking increases cross-community exposure over plain popularity (0.953 to 0.969) and clearly raises diversity and novelty, at the cost of roughly 20 % of recall (0.051 to 0.041) and 14 % of nDCG (0.032 to 0.028).
- Popularity is already close to the ceiling on cross-community exposure, so the exposure gain is small in absolute terms; the diversity and novelty gains are larger.
- The graph and text Hybrid retriever is the most "echo-like" (lowest diversity and cross-community exposure) and also the least accurate of the non-random methods.

A smaller-sample run of the sentence-transformer variant (`transformers_baselines.ipynb`, 2,502 graph users) gives Precision@10 / Recall@10 / nDCG@10 of 0.0057 / 0.0271 / 0.0160 for the graph and text retriever, against 0.0211 / 0.1080 / 0.0694 for popularity. It is not directly comparable to the table above because of the different sample.

## Notebooks and reading order

1. `EchoAware_FriendRecommender_Reddit_Final.ipynb` - the complete pipeline and the source of the results above. Start here.
2. `Data_Exploration_Visualization.ipynb` - dataset statistics, degree distribution and community plot.
3. `transformers_baselines.ipynb` - sentence-transformer text embeddings in place of the CNN, on a smaller sample.
4. `notebooks/exploration/` - earlier iterations kept for reference (baselines, CNN, Node2Vec-only, retrieval and ranking prototypes, visualisation). They read `data/processed/` relative to the working directory, so run them from a directory that contains it (for example by copying a notebook to the repository root).

## Running

```bash
conda env create -f environment.yml
conda activate escaping_echo
python -m ipykernel install --name escaping_echo --user   # if the kernel is not listed
jupyter notebook
```

Open the final notebook, select the `escaping_echo` kernel and run all cells.

- The first run streams the data from Hugging Face (network access required) and caches it under `data/processed/`.
- A fixed seed (`SEED = 42`) is set for Python, NumPy and PyTorch, and Node2Vec and Word2Vec use the same seed with a single worker. Exact reproducibility across machines and devices is not guaranteed.
- Runtime class: minutes to tens of minutes on a laptop. CNN training uses the Apple `mps` device if available and falls back to CPU; Node2Vec and the per-user evaluation loops are CPU-bound.
- `transformers_baselines.ipynb` additionally needs `sentence-transformers`, which is not part of `environment.yml` (`pip install sentence-transformers`).

## Limitations

- One subreddit, a 100,000-comment sample, one seed and no confidence intervals; absolute accuracy values are very small and differences between close methods should not be over-interpreted.
- Replies are a proxy for friendship: the evaluation measures prediction of future replies, not accepted friend suggestions.
- Louvain communities are computed on the same training graph that defines the cross-community metric, and the metric is not independent of the graph-based methods. Random recommendation scores high on it, so it must be read together with the accuracy metrics.
- The re-ranking penalty (0.5) and the Hybrid weight (0.5) are fixed; no sweep of the accuracy versus exposure trade-off is included.
- Text profiles use each user's ten most recent training comments, truncated to 256 tokens.
- Streamed data and trained models are not versioned; results depend on the dataset snapshot at download time.
