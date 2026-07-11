"""
rfm_analysis.py
RFM analysis + CLV + churn labeling on the REAL UCI Online Retail dataset.

Handles real-world data quality issues:
- ~25% of rows have missing CustomerID (guest/unlinked transactions) -- dropped,
  since RFM analysis is fundamentally customer-level
- Cancelled orders (InvoiceNo starting with 'C') -- dropped, they represent
  reversed transactions, not genuine purchases
- Non-positive Quantity/UnitPrice (returns, data entry errors) -- dropped

Input:  data/online_retail_raw.csv
Output: data/rfm_features.csv
"""

import pandas as pd
import numpy as np

CHURN_THRESHOLD_DAYS = 90  # chosen after checking churn-rate sensitivity (33% at 90 days -- a clean, realistic split)


def load_and_clean(path="data/online_retail_raw.csv"):
    df = pd.read_csv(path, encoding="utf-8-sig")

    before = len(df)

    # Drop cancelled orders (InvoiceNo starting with 'C' = credit/cancellation)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Drop rows with missing CustomerID -- can't attribute to a customer
    df = df[df["CustomerID"].notnull()]

    # Drop non-positive quantity/price (returns, free items, data errors)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    df["total_amount"] = df["Quantity"] * df["UnitPrice"]

    after = len(df)
    print(f"Cleaned {before:,} raw rows -> {after:,} valid rows ({after/before*100:.1f}% retained)")
    return df


def compute_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        frequency=("InvoiceNo", "nunique"),       # distinct orders, not line items
        monetary=("total_amount", "sum"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        country=("Country", "first"),
        n_unique_products=("StockCode", "nunique"),
    ).reset_index()

    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]
    rfm["tenure_days"] = (snapshot_date - rfm["first_purchase"]).dt.days
    rfm["purchase_rate"] = rfm["frequency"] / (rfm["tenure_days"] / 30).clip(lower=1)
    rfm["clv"] = (rfm["avg_order_value"] * rfm["purchase_rate"] * 12).round(2)

    rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    def segment(row):
        if row["RFM_score"] >= 13:
            return "Champions"
        elif row["RFM_score"] >= 10:
            return "Loyal Customers"
        elif row["RFM_score"] >= 7:
            return "Potential Loyalists"
        elif row["RFM_score"] >= 5:
            return "At Risk"
        else:
            return "Lost"

    rfm["segment"] = rfm.apply(segment, axis=1)
    rfm["churned"] = (rfm["recency"] > CHURN_THRESHOLD_DAYS).astype(int)

    return rfm


if __name__ == "__main__":
    df = load_and_clean()
    rfm = compute_rfm(df)
    rfm.to_csv("data/rfm_features.csv", index=False)

    print(f"\nRFM feature table: {len(rfm):,} unique customers")
    print(f"Churn rate: {rfm['churned'].mean()*100:.2f}%")
    print(f"\nSegment distribution:\n{rfm['segment'].value_counts()}")
    print(f"\nSample:\n{rfm.head()}")
