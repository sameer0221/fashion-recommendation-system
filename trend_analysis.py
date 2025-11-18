import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid")

# ---------------------------------------------------------
# Load metadata safely
# ---------------------------------------------------------
meta_df = pd.read_csv("data/styles.csv", on_bad_lines='skip')

# Basic cleaning
meta_df = meta_df.dropna(subset=["subCategory", "baseColour", "season"])

# Convert year to numeric
meta_df["year"] = pd.to_numeric(meta_df["year"], errors="coerce").fillna(0).astype(int)
meta_df = meta_df[meta_df["year"] > 2000]   # remove invalid 0 years

# ---------------------------------------------------------
# 1. Top 10 Subcategories Trend Over Years
# ---------------------------------------------------------
top_cats = meta_df['subCategory'].value_counts().nlargest(10).index
filtered_years = meta_df[meta_df['subCategory'].isin(top_cats)]

plt.figure(figsize=(12,6))
sns.countplot(data=filtered_years, x='year', hue='subCategory')
plt.title("Top 10 Fashion Categories Popularity Over Years")
plt.xticks(rotation=45)
plt.legend(title="SubCategory", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("category_popularity_years.png")
print("✅ Saved: category_popularity_years.png")

# ---------------------------------------------------------
# 2. Seasonal Trends of Top 10 Subcategories
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
sns.countplot(data=filtered_years, x='season', hue='subCategory')
plt.title("Top Categories Popularity Across Seasons")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("category_popularity_seasons.png")
print("✅ Saved: category_popularity_seasons.png")

# ---------------------------------------------------------
# 3. Top 10 Colors
# ---------------------------------------------------------
top_colors = meta_df['baseColour'].value_counts().nlargest(10).index
filtered_colors = meta_df[meta_df['baseColour'].isin(top_colors)]

plt.figure(figsize=(10,6))
sns.countplot(data=filtered_colors, x='baseColour', order=top_colors, palette="Set2")
plt.title("Top 10 Most Popular Fashion Colors")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_colors.png")
print("✅ Saved: top_colors.png")

# ---------------------------------------------------------
# 4. Category Distribution (Pie Chart)
# ---------------------------------------------------------
plt.figure(figsize=(7,7))
meta_df['masterCategory'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Overall Master Category Distribution")
plt.ylabel("")
plt.savefig("category_distribution.png")
print("✅ Saved: category_distribution.png")

# ---------------------------------------------------------
# 5. Tshirts Popularity Over Years
# ---------------------------------------------------------
tshirts = meta_df[meta_df['subCategory'] == "Tshirts"]

if len(tshirts) > 0:
    plt.figure(figsize=(8,5))
    sns.countplot(data=tshirts, x='year')
    plt.title("Tshirts Popularity Trend Over Years")
    plt.tight_layout()
    plt.savefig("tshirts_trend.png")
    print("✅ Saved: tshirts_trend.png")
else:
    print("⚠ No Tshirts category found in dataset")

# ---------------------------------------------------------
# 6. Color Trends Heatmap (Year vs Color)
# ---------------------------------------------------------
color_year = pd.crosstab(meta_df['year'], meta_df['baseColour'])

plt.figure(figsize=(12,6))
sns.heatmap(color_year[top_colors], cmap="YlGnBu")
plt.title("Top Color Trends Over Years")
plt.tight_layout()
plt.savefig("color_trends.png")
print("✅ Saved: color_trends.png")

# ---------------------------------------------------------
# 7. Price Analysis (IF price column exists)
# ---------------------------------------------------------
if "price" in meta_df.columns:
    plt.figure(figsize=(10,6))
    sns.boxplot(data=meta_df, x='masterCategory', y='price')
    plt.title("Price Range Distribution by Master Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("price_distribution.png")
    print("✅ Saved: price_distribution.png")

# ---------------------------------------------------------
# 8. Rating Analysis (IF rating column exists)
# ---------------------------------------------------------
if "rating" in meta_df.columns:
    plt.figure(figsize=(10,5))
    sns.histplot(meta_df['rating'].dropna(), kde=True)
    plt.title("Rating Distribution Across Products")
    plt.tight_layout()
    plt.savefig("rating_distribution.png")
    print("✅ Saved: rating_distribution.png")

print("\n🎉 All analysis images generated successfully!")
