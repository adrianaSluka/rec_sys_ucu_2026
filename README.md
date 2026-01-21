# Book Crossing Recommendation System

A comprehensive recommendation system project using the Book Crossing dataset, implementing classical recommendation paradigms including similarity-based recommenders and matrix factorization methods.

## Project Overview

This project is developed as part of a Recommendation Systems course capstone project. The goal is to build, evaluate, and compare classical recommendation algorithms using a well-defined offline evaluation protocol.

### Objectives
- Implement similarity-based recommenders (user-based and item-based collaborative filtering)
- Implement matrix factorization methods (SVD, ALS)
- Establish robust offline evaluation methodology
- Analyze and document data pathologies and their impact on recommendations

## Repository Structure

```
ucu-recs-system/
├── data/
│   ├── raw/                    # Raw dataset files (not tracked in git)
│   │   ├── BX-Book-Ratings.csv
│   │   ├── BX-Books.csv
│   │   └── BX-Users.csv
│   └── processed/              # Preprocessed data files
├── models/                     # Trained model artifacts
├── evaluation/
│   └── EVALUATION_METHODOLOGY.md  # Evaluation design document
├── experiments/                # Experiment outputs and visualizations
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── EDA_SUMMARY.md         # EDA findings and modeling implications
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading utilities
│   │   └── splitter.py        # Temporal train/val/test splitting
│   ├── models/
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # RMSE, NDCG, Precision, etc.
│   │   └── pipeline.py        # Evaluation pipeline & baselines
│   └── utils/
│       └── __init__.py
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ucu-recs-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

The Book Crossing dataset needs to be downloaded manually due to the original source being unavailable.

**Option 1: Kaggle (Recommended)**
1. Visit: https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset
2. Download the dataset
3. Extract the CSV files to `data/raw/`

**Option 2: GroupLens**
1. Visit: https://grouplens.org/datasets/book-crossing/
2. Follow download instructions
3. Extract to `data/raw/`

Required files in `data/raw/`:
- `BX-Book-Ratings.csv`
- `BX-Books.csv`
- `BX-Users.csv`

### 5. Verify Installation

```bash
python -c "from src.data.loader import load_all_data; print('Setup successful!')"
```

## Running the EDA

### Option 1: Jupyter Notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Option 2: JupyterLab

```bash
jupyter lab
# Navigate to notebooks/01_eda.ipynb
```

## Dataset Information

The **Book-Crossing Dataset** was collected by Cai-Nicolas Ziegler in August/September 2004 from the Book-Crossing community.

### Dataset Statistics
| Component | Count |
|-----------|-------|
| Ratings | ~1,149,780 |
| Books | ~271,379 |
| Users | ~278,858 |

### Rating Schema
- **0**: Implicit feedback (interaction without explicit rating)
- **1-10**: Explicit rating scale

### Key Data Characteristics
- **Extreme sparsity**: ~99.99%
- **Strong popularity bias**: Top 1% of books receive >30% of ratings
- **Severe cold-start**: 85% of books have ≤5 ratings
- **Mixed feedback**: 62% implicit, 38% explicit

## Project Deliverables

### Phase 1: Repository Setup and EDA (Current)
- [x] Repository structure with clear organization
- [x] Reproducible environment setup
- [x] Comprehensive EDA notebook
- [x] EDA summary with modeling implications
- [x] Data pathology identification

### Phase 2: Similarity-Based Recommenders (Upcoming)
- [ ] User-based collaborative filtering
- [ ] Item-based collaborative filtering
- [ ] Similarity metrics comparison

### Phase 3: Matrix Factorization (Upcoming)
- [ ] SVD implementation
- [ ] ALS implementation
- [ ] Implicit feedback handling

### Phase 4: Offline Evaluation Strategy (Complete)
- [x] Temporal split strategy (by publication year)
- [x] Metric implementation (RMSE, MAE, NDCG, Precision, Recall, Hit Rate)
- [x] Evaluation pipeline with baseline models
- [x] Evaluation methodology document

## Key Findings from EDA

1. **Sparsity Challenge**: The dataset's 99.99% sparsity requires careful algorithm selection and strong regularization.

2. **Popularity Skew**: Severe long-tail distribution means popularity-based baselines will be strong competitors.

3. **Cold-Start Problem**: Most users and items have very few interactions, limiting collaborative filtering effectiveness.

4. **Implicit Feedback Dominance**: 62% of interactions are implicit, suggesting hybrid or implicit-feedback-specific methods.

See [EDA_SUMMARY.md](notebooks/EDA_SUMMARY.md) for detailed findings and recommendations.

## Usage Examples

### Loading Data

```python
from src.data.loader import load_all_data, load_ratings

# Load all datasets
ratings, books, users = load_all_data('data/raw')

# Load only ratings
ratings = load_ratings('data/raw')
```

### Basic Statistics

```python
print(f"Total ratings: {len(ratings):,}")
print(f"Unique users: {ratings['user_id'].nunique():,}")
print(f"Unique books: {ratings['isbn'].nunique():,}")
```

### Creating Temporal Split

```python
from src.data.splitter import create_temporal_split, SplitConfig

config = SplitConfig(
    train_max_year=1999,      # Train: ≤1999
    val_max_year=2001,        # Val: 2000-2001, Test: 2002-2004
    min_user_interactions=5,
    min_item_interactions=5,
    explicit_only=True
)

train_df, val_df, test_df, split_info = create_temporal_split(ratings, books, config)
```

### Running Baseline Evaluation

```python
from src.evaluation import run_baseline_evaluation

results = run_baseline_evaluation(train_df, test_df, k_values=[5, 10, 20])
```

## References

1. Ziegler, C. N., McNee, S. M., Konstan, J. A., & Lausen, G. (2005). Improving recommendation lists through topic diversification. *Proceedings of the 14th international conference on World Wide Web*.

2. Book-Crossing Dataset: https://grouplens.org/datasets/book-crossing/

## License

This project is for educational purposes as part of a university course.

## Author

Developed for UCU Recommendation Systems Course Capstone Project.
