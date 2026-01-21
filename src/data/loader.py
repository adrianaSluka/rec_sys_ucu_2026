"""
Data loading utilities for Book Crossing dataset.
"""

import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def download_book_crossing_data(data_dir: str = "data/raw") -> None:
    """
    Download Book Crossing dataset.

    The dataset can be obtained from:
    - Kaggle: https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset
    - GroupLens: https://grouplens.org/datasets/book-crossing/

    Args:
        data_dir: Directory to save the raw data files
    """
    # Try multiple URLs as the original source may be unavailable
    urls = [
        "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip",
        "https://github.com/caserec/Datasets-for-Recommender-Systems/raw/master/Processed%20Datasets/BookCrossing/BX-CSV-Dump.zip",
    ]

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    zip_path = data_path / "BX-CSV-Dump.zip"

    # Check if CSV files already exist
    csv_files = list(data_path.glob("BX-*.csv"))
    if len(csv_files) >= 3:
        print("CSV files already present")
        return

    if zip_path.exists():
        print(f"Dataset already downloaded at {zip_path}")
    else:
        downloaded = False
        for url in urls:
            try:
                print(f"Attempting download from {url}...")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(zip_path, 'wb') as f:
                    with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))

                print(f"Downloaded to {zip_path}")
                downloaded = True
                break
            except (requests.exceptions.RequestException, Exception) as e:
                print(f"Failed to download from {url}: {e}")
                continue

        if not downloaded:
            print("\n" + "="*60)
            print("MANUAL DOWNLOAD REQUIRED")
            print("="*60)
            print("Please download the Book Crossing dataset manually:")
            print("1. Go to: https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset")
            print("2. Download the dataset")
            print("3. Extract the CSV files to: data/raw/")
            print("   Required files: BX-Books.csv, BX-Book-Ratings.csv, BX-Users.csv")
            print("="*60)
            return

    # Extract files
    csv_files = list(data_path.glob("BX-*.csv"))
    if len(csv_files) >= 3:
        print("CSV files already extracted")
    else:
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("Extraction complete")


def load_ratings(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the Book-Ratings dataset.

    Supports both naming conventions:
    - BX-Book-Ratings.csv (original format)
    - Ratings.csv (Kaggle format)

    Args:
        data_dir: Directory containing the raw data files

    Returns:
        DataFrame with columns: user_id, isbn, rating
    """
    data_path = Path(data_dir)

    # Try different file names
    possible_names = ["BX-Book-Ratings.csv", "Ratings.csv"]
    ratings_path = None
    for name in possible_names:
        candidate = data_path / name
        if candidate.exists():
            ratings_path = candidate
            break

    if ratings_path is None:
        raise FileNotFoundError(
            f"Ratings file not found. Expected one of {possible_names} in {data_dir}"
        )

    ratings = pd.read_csv(
        ratings_path,
        sep=';',
        encoding='latin-1',
        quotechar='"',
        escapechar='\\',
        on_bad_lines='skip'
    )

    # Standardize column names
    ratings.columns = ['user_id', 'isbn', 'rating']
    ratings['user_id'] = ratings['user_id'].astype(str)
    ratings['isbn'] = ratings['isbn'].astype(str).str.strip()
    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')

    return ratings


def load_books(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the Books dataset.

    Supports both naming conventions:
    - BX-Books.csv (original format with image URLs)
    - Books.csv (Kaggle format without image URLs)

    Args:
        data_dir: Directory containing the raw data files

    Returns:
        DataFrame with book information
    """
    data_path = Path(data_dir)

    # Try different file names
    possible_names = ["BX-Books.csv", "Books.csv"]
    books_path = None
    for name in possible_names:
        candidate = data_path / name
        if candidate.exists():
            books_path = candidate
            break

    if books_path is None:
        raise FileNotFoundError(
            f"Books file not found. Expected one of {possible_names} in {data_dir}"
        )

    books = pd.read_csv(
        books_path,
        sep=';',
        encoding='latin-1',
        quotechar='"',
        escapechar='\\',
        on_bad_lines='skip',
        low_memory=False
    )

    # Handle different column formats
    if len(books.columns) == 8:
        # Original format with image URLs
        books.columns = [
            'isbn', 'title', 'author', 'year', 'publisher',
            'image_url_s', 'image_url_m', 'image_url_l'
        ]
    elif len(books.columns) == 5:
        # Kaggle format without image URLs
        books.columns = ['isbn', 'title', 'author', 'year', 'publisher']
    else:
        # Try to use original column names, standardize first column
        original_cols = list(books.columns)
        books.columns = ['isbn', 'title', 'author', 'year', 'publisher'] + original_cols[5:]

    books['isbn'] = books['isbn'].astype(str).str.strip()
    books['year'] = pd.to_numeric(books['year'], errors='coerce')

    return books


def load_users(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the Users dataset.

    Supports both naming conventions:
    - BX-Users.csv (original format with location)
    - Users.csv (Kaggle format, may have different columns)

    Args:
        data_dir: Directory containing the raw data files

    Returns:
        DataFrame with user information
    """
    data_path = Path(data_dir)

    # Try different file names
    possible_names = ["BX-Users.csv", "Users.csv"]
    users_path = None
    for name in possible_names:
        candidate = data_path / name
        if candidate.exists():
            users_path = candidate
            break

    if users_path is None:
        raise FileNotFoundError(
            f"Users file not found. Expected one of {possible_names} in {data_dir}"
        )

    users = pd.read_csv(
        users_path,
        sep=';',
        encoding='latin-1',
        quotechar='"',
        escapechar='\\',
        on_bad_lines='skip',
        low_memory=False
    )

    # Handle different column formats
    if len(users.columns) == 3:
        users.columns = ['user_id', 'location', 'age']
    elif len(users.columns) == 2:
        # Kaggle format without location
        users.columns = ['user_id', 'age']
        users['location'] = None
    else:
        # Use first column as user_id, try to identify others
        original_cols = list(users.columns)
        users.columns = ['user_id'] + original_cols[1:]
        if 'age' not in users.columns and 'Age' not in users.columns:
            users['age'] = None
        if 'location' not in users.columns and 'Location' not in users.columns:
            users['location'] = None

    users['user_id'] = users['user_id'].astype(str)
    users['age'] = pd.to_numeric(users['age'], errors='coerce')

    return users


def load_all_data(data_dir: str = "data/raw") -> tuple:
    """
    Load all three datasets.

    Args:
        data_dir: Directory containing the raw data files

    Returns:
        Tuple of (ratings, books, users) DataFrames
    """
    ratings = load_ratings(data_dir)
    books = load_books(data_dir)
    users = load_users(data_dir)

    return ratings, books, users
