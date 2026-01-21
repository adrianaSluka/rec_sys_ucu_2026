# Exploratory Data Analysis Summary: Book Crossing Dataset

## Executive Summary

This document summarizes the key findings from the exploratory data analysis of the Book Crossing dataset and provides explicit recommendations for modeling and evaluation strategies.

## Dataset Overview

The Book Crossing dataset was collected by Cai-Nicolas Ziegler in August-September 2004 from the Book-Crossing community. It contains:

| Component | Count | Description |
|-----------|-------|-------------|
| Ratings | ~1.15M | User-book interactions |
| Books | ~271K | Unique ISBN entries |
| Users | ~278K | Anonymized user profiles |

### Rating Schema
- **0**: Implicit feedback (user interacted but didn't provide explicit rating)
- **1-10**: Explicit rating scale

## Key Findings

### 1. Interaction Sparsity

**Finding**: The dataset exhibits extreme sparsity.

| Metric | All Interactions | Explicit Only |
|--------|-----------------|---------------|
| Density | ~0.0015% | ~0.0006% |
| Sparsity | ~99.9985% | ~99.9994% |

**Implication**: Standard matrix factorization methods will struggle. Regularization is critical. Consider implicit feedback methods.

### 2. Implicit vs Explicit Feedback Split

| Type | Count | Percentage |
|------|-------|------------|
| Implicit (rating=0) | ~716K | ~62% |
| Explicit (rating 1-10) | ~434K | ~38% |

**Implication**:
- Cannot treat all interactions equally
- Implicit feedback provides additional signal but different semantics
- Consider hybrid approaches or separate models

### 3. User Activity Distribution

| Metric | Value |
|--------|-------|
| Mean ratings/user | ~4.1 |
| Median ratings/user | 1-2 |
| Users with ≤5 ratings | ~62% |
| Users with ≤10 ratings | ~75% |
| Gini coefficient | ~0.85 |

**Implication**: Severe user cold-start problem. Most users have very few interactions.

### 4. Item (Book) Popularity Distribution

| Metric | Value |
|--------|-------|
| Mean ratings/item | ~4.2 |
| Median ratings/item | 1 |
| Items with ≤5 ratings | ~85% |
| Items with 1 rating | ~60% |
| Gini coefficient | ~0.90 |

**Finding**: Extreme long-tail distribution. Top 1% of books account for >30% of all ratings.

**Implication**:
- Strong popularity baseline will be hard to beat
- Most items have insufficient data for reliable CF
- Popularity debiasing needed for fair evaluation

### 5. Temporal Dynamics

Since the dataset lacks explicit timestamps, analysis was performed on publication years:
- Strong recency bias: Books from 1995-2004 dominate
- Classic literature (pre-1980) is underrepresented
- Peak ratings around 2000-2002 publications

**Implication**: Model should account for recency; temporal evaluation split may not be directly applicable.

### 6. Data Pathologies Identified

#### Pathology 1: Extreme Popularity Skew
- Power law distribution with α ≈ 1.0-1.2
- Top 5% items capture >50% of interactions
- Recommendation algorithms may degenerate to popularity lists

#### Pathology 2: Severe Cold-Start
- **User cold-start**: 62% of users have ≤5 ratings
- **Item cold-start**: 85% of items have ≤5 ratings
- New items/users have essentially no collaborative signal

#### Pathology 3: Rating Bias
- Users show strong personal bias in rating patterns
- Some users consistently rate high (>8), others low (<5)
- Average rating ~7.6 (skewed positive)

#### Pathology 4: Implicit/Explicit Imbalance
- 62% of interactions are implicit (rating=0)
- Users providing implicit feedback may have different behavior patterns

## Modeling Recommendations

### Data Preprocessing Pipeline

```
1. Load raw data
2. Separate implicit (rating=0) and explicit (rating>0) interactions
3. For explicit ratings:
   - Apply user mean normalization
   - Filter users with <5 ratings (for evaluation reliability)
   - Filter items with <10 ratings (for stable representations)
4. For implicit feedback:
   - Convert to binary (1 for any interaction)
   - Consider confidence weighting based on frequency
```

### Recommended Filtering Strategy

| Variant | User Threshold | Item Threshold | Purpose |
|---------|---------------|----------------|---------|
| Full | 1 | 1 | Maximum coverage |
| Standard | 5 | 10 | Balanced approach |
| Dense | 10 | 20 | Reliable evaluation |

### Algorithm Selection Guide

| Algorithm | Suitability | Notes |
|-----------|-------------|-------|
| Popularity Baseline | High | Strong baseline, use as comparison |
| User-based CF | Low | Too many users, sparse profiles |
| Item-based CF | Medium | Fewer items, but still sparse |
| SVD/ALS | Medium | Requires heavy regularization, low rank |
| SVD++ | Medium-High | Can leverage implicit feedback |
| BPR | High | Designed for implicit feedback |
| Neural CF | Medium | Requires careful regularization |

### Hyperparameter Starting Points

For Matrix Factorization:
- **Latent factors**: Start with k=20-50 (sparsity limits higher)
- **Regularization**: λ = 0.01-0.1 (strong regularization needed)
- **Learning rate**: 0.001-0.01 for SGD
- **Iterations**: 20-50 epochs with early stopping

### Evaluation Protocol

1. **Split Strategy**:
   - Random 80/10/10 train/validation/test
   - Leave-one-out for ranking evaluation
   - Stratify by user activity level

2. **Metrics**:
   - **Rating prediction**: RMSE, MAE (explicit only)
   - **Ranking**: NDCG@K, Precision@K, Recall@K, MAP
   - **Coverage**: Catalog coverage, user coverage
   - **Beyond accuracy**: Novelty, diversity

3. **Baselines**:
   - Global mean
   - User mean
   - Item mean
   - Most popular
   - Random

4. **Cold-start evaluation**:
   - Report metrics separately for:
     - Warm users (>20 ratings) vs cold users (5-20 ratings)
     - Warm items (>50 ratings) vs cold items (10-50 ratings)

## Summary Table: Dataset Characteristics

| Characteristic | Value | Severity | Impact on Modeling |
|---------------|-------|----------|-------------------|
| Sparsity | 99.99% | Critical | Use implicit feedback methods |
| User cold-start | 62% have ≤5 | Severe | Filter or use content features |
| Item cold-start | 85% have ≤5 | Severe | Use metadata, popularity fallback |
| Popularity skew | Gini=0.90 | High | Popularity debiasing in eval |
| Rating bias | Users vary 5-9 mean | Moderate | User normalization |
| Implicit ratio | 62% | Notable | Separate or hybrid models |

## Files Generated

- `notebooks/01_eda.ipynb` - Full EDA notebook with visualizations
- `experiments/eda_*.png` - Generated visualizations
- `experiments/eda_summary_stats.csv` - Summary statistics table

## References

1. Ziegler, C. N., et al. (2005). Improving recommendation lists through topic diversification. WWW '05.
2. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. ICDM '08.
3. Koren, Y. (2008). Factorization meets the neighborhood: a multifaceted collaborative filtering model. KDD '08.
