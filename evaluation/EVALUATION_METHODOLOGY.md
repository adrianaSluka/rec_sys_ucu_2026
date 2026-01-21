# Offline Evaluation Methodology

## 1. Overview

This document defines the offline evaluation methodology for the Book Crossing recommendation system. The methodology is designed to assess recommendation quality under realistic conditions while acknowledging the limitations of offline evaluation.

## 2. Data Split Strategy

### 2.1 Temporal Split Rationale

We employ a **temporal split** based on book publication year rather than random splitting. This choice is motivated by:

1. **Realism**: In production, models are trained on historical data and must predict future user preferences. Random splits create data leakage where future information aids past predictions.

2. **Cold-Start Simulation**: New books (recent publications) naturally have fewer ratings. Temporal splitting forces the model to handle newer, less-rated items.

3. **Practical Validity**: Publishing industry patterns matter—a model that only recommends pre-2000 books would be commercially unviable.

### 2.2 Split Mechanics

Since the Book Crossing dataset (collected in 2004) lacks explicit timestamps, we use **publication year** as a temporal proxy:

| Split | Publication Year | Purpose |
|-------|-----------------|---------|
| **Train** | ≤ 1999 | Historical data for model training |
| **Validation** | 2000-2001 | Hyperparameter tuning |
| **Test** | 2002-2004 | Final evaluation (held out) |

This creates a realistic scenario: training on "older" books and evaluating on "newer" releases.

### 2.3 Data Filtering

Before splitting, we apply filtering to ensure evaluation reliability:

1. **Explicit ratings only**: Remove implicit feedback (rating=0) since we evaluate rating prediction quality
2. **Minimum user activity**: ≥5 ratings per user (users with fewer ratings provide unreliable signal)
3. **Minimum item activity**: ≥5 ratings per item (items need sufficient data for CF)
4. **Valid publication years**: 1900-2004 (filter data entry errors)

### 2.4 User Overlap Constraint

Critical requirement: **Test users must exist in training data**. We can only evaluate collaborative filtering for users with some historical behavior. Users appearing only in test set are excluded from evaluation.

## 3. Evaluation Tasks

We evaluate models on two complementary tasks:

### 3.1 Rating Prediction

**Goal**: Predict the exact rating a user would give to an item.

**Use case**: Estimating user satisfaction, sorting candidates by predicted preference.

### 3.2 Top-N Ranking

**Goal**: Generate an ordered list of N items the user would find relevant.

**Use case**: Homepage recommendations, "You might like" sections.

**Relevance definition**: Items with rating ≥ 6 (on 1-10 scale) are considered "relevant."

## 4. Metrics

### 4.1 Primary Metrics

| Task | Primary Metric | Justification |
|------|---------------|---------------|
| Rating Prediction | **RMSE** | Standard metric, penalizes large errors quadratically |
| Ranking | **NDCG@10** | Position-aware, handles graded relevance, normalized |

**RMSE (Root Mean Squared Error)**:
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(r_i - \hat{r}_i)^2}$$

**NDCG@K (Normalized Discounted Cumulative Gain)**:
$$NDCG@K = \frac{DCG@K}{IDCG@K}, \quad DCG@K = \sum_{i=1}^{K}\frac{rel_i}{\log_2(i+1)}$$

### 4.2 Secondary/Diagnostic Metrics

| Metric | Purpose |
|--------|---------|
| **MAE** | Interpretable rating error (average deviation in rating points) |
| **Hit Rate@10** | Binary success metric: did we recommend at least one relevant item? |
| **Precision@10** | Fraction of recommendations that are relevant |
| **Recall@10** | Fraction of relevant items that were recommended |
| **Catalog Coverage@20** | Diversity: what fraction of items ever get recommended? |

### 4.3 Metric Selection Justification

1. **RMSE over MAE** as primary: RMSE penalizes large errors more, which is important for user satisfaction (a prediction off by 4 points is much worse than two predictions off by 2 points each).

2. **NDCG over Precision/Recall** as primary: NDCG accounts for position (relevant items ranked higher score better) and is normalized, enabling comparison across users with different numbers of relevant items.

3. **Coverage as diagnostic**: Ensures the model doesn't degenerate into recommending only a few popular items to everyone.

## 5. Evaluation Protocol

### 5.1 Rating Prediction Protocol

```
For each (user, item, rating) in test set:
    1. Predict rating using model
    2. Compute squared error
Aggregate: RMSE = sqrt(mean(squared_errors))
```

### 5.2 Ranking Protocol

```
For each user in test set with ≥1 relevant item:
    1. Identify relevant items (rating ≥ 6 in test)
    2. Identify items to exclude (user's training items)
    3. Generate top-N recommendations
    4. Compute precision, recall, NDCG, hit rate
Aggregate: Mean across all evaluated users
```

### 5.3 Baseline Models

Every model must be compared against these baselines:

| Baseline | Description |
|----------|-------------|
| Global Mean | Predicts dataset average rating |
| User Mean | Predicts user's average rating |
| Item Mean | Predicts item's average rating |
| Popularity | Recommends most-rated items |

## 6. What This Evaluation Captures

### 6.1 Strengths

1. **Predictive accuracy**: How well the model estimates user preferences
2. **Ranking quality**: Whether relevant items appear at top of recommendations
3. **Temporal generalization**: Performance on newer items not seen during training
4. **Cold-start handling**: Implicit evaluation of performance on less-rated items

### 6.2 Captured Scenarios

- Users rating newly published books
- Model's ability to generalize beyond training item set
- Relative comparison between algorithms under controlled conditions

## 7. What This Evaluation Fails to Capture

### 7.1 Fundamental Limitations

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Selection Bias** | Users rate items they chose to read; we can't measure performance on items users would never consider | Acknowledged; consider propensity-weighted evaluation in future work |
| **Missing Feedback** | Users who disliked a book might not rate it at all | Use implicit feedback data for broader signal |
| **Temporal Dynamics** | User preferences change over time; our proxy (publication year) is imperfect | Accept as limitation of this dataset |
| **Context** | Recommendations in practice depend on context (mood, time, device) | Not captured; static evaluation |

### 7.2 Business Relevance Gaps

1. **User Engagement**: Offline accuracy ≠ click-through rate or read completion
2. **Serendipity**: We measure relevance but not discovery of unexpected gems
3. **Diversity**: Users may want variety, not just accuracy
4. **Freshness**: Production systems must balance quality with recency

### 7.3 Dataset-Specific Issues

1. **No real timestamps**: Publication year is a proxy; actual rating time unknown
2. **2004 data cutoff**: Models trained here won't reflect modern reading patterns
3. **Implicit ratings semantics**: Rating=0 may mean different things (browsed, owned, started reading)

## 8. Implementation Notes

### 8.1 Code Location

```
src/
├── data/
│   └── splitter.py      # Temporal split implementation
└── evaluation/
    ├── metrics.py       # Metric implementations
    └── pipeline.py      # Evaluation pipeline & baselines
```

### 8.2 Usage Example

```python
from src.data.loader import load_all_data
from src.data.splitter import create_temporal_split, SplitConfig
from src.evaluation import EvaluationPipeline, run_baseline_evaluation

# Load data
ratings, books, users = load_all_data('data/raw')

# Create temporal split
config = SplitConfig(
    train_max_year=1999,
    val_max_year=2001,
    min_user_interactions=5,
    min_item_interactions=5,
    explicit_only=True
)
train_df, val_df, test_df, split_info = create_temporal_split(ratings, books, config)

# Run baseline evaluation
results = run_baseline_evaluation(train_df, test_df)
```

## 9. Summary

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Split | Temporal (by publication year) | Simulates real deployment scenario |
| Primary metric (rating) | RMSE | Standard, penalizes large errors |
| Primary metric (ranking) | NDCG@10 | Position-aware, normalized |
| Relevance threshold | Rating ≥ 6 | Captures genuinely positive feedback |
| Minimum activity | 5 ratings per user/item | Ensures reliable evaluation signal |

This methodology provides a principled, reproducible evaluation framework while explicitly acknowledging its limitations. Results should be interpreted as relative comparisons between methods under these specific conditions, not as absolute performance guarantees.
