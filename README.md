# Myntra Fashion Catalog Segmentation

**Segmenting a 526K-product fashion catalog into four actionable groups — including 18K "risky" products that discount heavily and still rate poorly.**

EDA and K-Means clustering on a Myntra product-listings dataset (526,564 rows; 117,565 products after cleaning), covering pricing, discounts, ratings, and reviews across categories and brands.

## The segmentation (K=4, chosen by elbow method)

Clustered on scaled price, discount %, rating, and review count:

| Segment | Products | Profile | What to do with it |
|---|---|---|---|
| Budget Deals | 60,304 | ~₹785 avg price, ~60% discount, 4.2★ | Volume engine — keep stocked, thin margins |
| Bestsellers | 5,842 | 519 avg reviews (9× catalog average), 4.1★ | Protect availability & search placement |
| Premium | 33,100 | ~₹1,506 avg price, lowest discount (32%), best ratings (4.3★) | Margin engine — resist discounting |
| **Risky** | 18,319 | **3.26★ despite ~56% discount** | Heavy discounts aren't fixing bad products — review or delist |

The "Risky" cluster is the actionable finding: 16% of the catalog is being discounted hard and still rates worst. Discounting is masking a quality problem, not solving it.

## Other findings

- Top category by review volume: Indian Wear; top brand within it: Anouk.
- Price-tier breakdown (Budget/Mid-range/Premium/Luxury) and top-rated products computed with minimum-review thresholds to filter low-sample noise.

## Data honesty notes

- The `Reviews` field in the source data is **capped at 999**, and review count ≠ sales. The notebook's `Revenue = DiscountPrice × Reviews` column is a *relative popularity proxy* for ranking products — the absolute rupee totals it produces are not real revenue and are not reported here as findings.
- `myn.csv` is not committed (too large); it's the publicly available Myntra products dataset (e.g. on Kaggle). Place it next to the notebook to reproduce.

## Run

```
pip install -r requirements.txt
jupyter notebook main.ipynb
```

## Tech stack

Python · pandas · NumPy · scikit-learn (StandardScaler, KMeans) · matplotlib · seaborn
