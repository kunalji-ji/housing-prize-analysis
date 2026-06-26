import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("Housing Price Prediction - German Metropolitan Areas")
print("=" * 60)

# ── 1. DATA GENERATION ──────────────────────────────────────

np.random.seed(42)
N = 2000

cities = ['Berlin', 'Munich', 'Frankfurt', 'Hamburg',
          'Cologne', 'Stuttgart', 'Düsseldorf', 'Leipzig']

city_base_price = {
    'Berlin': 5200, 'Munich': 9800, 'Frankfurt': 7100,
    'Hamburg': 6800, 'Cologne': 5500, 'Stuttgart': 6200,
    'Düsseldorf': 5900, 'Leipzig': 3400
}
city_weights = [0.20, 0.18, 0.14, 0.14, 0.12, 0.08, 0.08, 0.06]

city_col      = np.random.choice(cities, size=N, p=city_weights)
size_sqm      = np.random.normal(75, 25, N).clip(25, 250)
rooms         = np.round(size_sqm / 30).clip(1, 8).astype(int)
age_years     = np.random.choice(range(0, 80), N)
floor         = np.random.randint(0, 12, N)
has_balcony   = np.random.binomial(1, 0.55, N)
has_parking   = np.random.binomial(1, 0.40, N)
has_elevator  = (floor > 3).astype(int)
distance_cbd  = np.random.exponential(8, N).clip(0.5, 40)
district_score = np.random.uniform(1, 10, N)

base = np.array([city_base_price[c] for c in city_col])
price_per_sqm = (
    base
    + district_score * 180
    - age_years * 22
    + has_balcony * 300
    + has_parking * 400
    + has_elevator * 200
    - distance_cbd * 80
    + floor * 50
    + np.random.normal(0, 600, N)
).clip(1500, 18000)

rent_per_sqm = (price_per_sqm / 280).clip(6, 45)

total_price = (price_per_sqm * size_sqm).round(-2)
monthly_rent = (rent_per_sqm * size_sqm).round()

df = pd.DataFrame({
    'city': city_col,
    'size_sqm': size_sqm.round(1),
    'rooms': rooms,
    'age_years': age_years,
    'floor': floor,
    'has_balcony': has_balcony,
    'has_parking': has_parking,
    'has_elevator': has_elevator,
    'distance_cbd_km': distance_cbd.round(2),
    'district_score': district_score.round(2),
    'price_per_sqm': price_per_sqm.round(2),
    'total_price_eur': total_price,
    'monthly_rent_eur': monthly_rent
})

print(f"\n[OK] Dataset created: {len(df)} listings across {df['city'].nunique()} cities")
print(df.describe().round(2))

# ── 2. EXPLORATORY DATA ANALYSIS ────────────────────────────
print("\n[EDA] Generating plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis - German Housing Market', fontsize=15, fontweight='bold')

city_avg = df.groupby('city')['price_per_sqm'].mean().sort_values(ascending=False)
axes[0,0].bar(city_avg.index, city_avg.values, color=plt.cm.Blues_r(np.linspace(0.3,0.8,len(city_avg))))
axes[0,0].set_title('Avg Price/sqm by City (EUR )')
axes[0,0].set_xlabel('City')
axes[0,0].set_ylabel('EUR  per sqm')
axes[0,0].tick_params(axis='x', rotation=30)
for i, v in enumerate(city_avg.values):
    axes[0,0].text(i, v+50, f'EUR {v:,.0f}', ha='center', fontsize=8)

axes[0,1].hist(df['price_per_sqm'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[0,1].axvline(df['price_per_sqm'].mean(), color='red', linestyle='--', label=f'Mean: EUR {df["price_per_sqm"].mean():,.0f}')
axes[0,1].set_title('Distribution of Price per sqm (EUR )')
axes[0,1].set_xlabel('EUR  per sqm')
axes[0,1].set_ylabel('Count')
axes[0,1].legend()

sample = df.sample(400, random_state=42)
scatter = axes[0,2].scatter(sample['size_sqm'], sample['price_per_sqm'],
                             c=sample['district_score'], cmap='RdYlGn', alpha=0.6, s=20)
axes[0,2].set_title('Size vs Price/sqm (colour = district score)')
axes[0,2].set_xlabel('Size (sqm)')
axes[0,2].set_ylabel('EUR  per sqm')
plt.colorbar(scatter, ax=axes[0,2], label='District Score')

age_bins = pd.cut(df['age_years'], bins=[0,5,15,30,50,80], labels=['0-5y','6-15y','16-30y','31-50y','50+y'])
age_price = df.groupby(age_bins)['price_per_sqm'].mean()
axes[1,0].bar(age_price.index.astype(str), age_price.values, color='coral')
axes[1,0].set_title('Avg Price/sqm by Property Age')
axes[1,0].set_xlabel('Age Band')
axes[1,0].set_ylabel('EUR  per sqm')

num_cols = ['size_sqm','rooms','age_years','floor','has_balcony','has_parking',
            'distance_cbd_km','district_score','price_per_sqm']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=axes[1,1], mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, square=True, linewidths=0.5, annot_kws={'size':7})
axes[1,1].set_title('Feature Correlation Matrix')
axes[1,1].tick_params(axis='x', rotation=45)

city_order = df.groupby('city')['monthly_rent_eur'].median().sort_values(ascending=False).index
df_plot = df[df['city'].isin(city_order)]
bp_data = [df[df['city']==c]['monthly_rent_eur'].values for c in city_order]
axes[1,2].boxplot(bp_data, tick_labels=city_order, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='navy'),
                  medianprops=dict(color='red', linewidth=2))
axes[1,2].set_title('Monthly Rent Distribution by City (EUR )')
axes[1,2].set_xlabel('City')
axes[1,2].set_ylabel('EUR /month')
axes[1,2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('outputs/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved outputs/eda_plots.png")

# ── 3. FEATURE ENGINEERING & PREPROCESSING ──────────────────
print("\n[ML] Preparing features...")

le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

features = ['city_encoded','size_sqm','rooms','age_years','floor',
            'has_balcony','has_parking','has_elevator',
            'distance_cbd_km','district_score']
target = 'price_per_sqm'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. MODEL TRAINING ────────────────────────────────────────
print("\n[ML] Training models...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=10),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42)
}

results = {}
for name, model in models.items():
    if 'Regression' in name:
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'preds': preds}
    print(f"  {name:25s} | MAE: EUR {mae:,.0f} | RMSE: EUR {rmse:,.0f} | R2: {r2:.4f}")

# ── 5. MODEL COMPARISON PLOT ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

names = list(results.keys())
maes  = [results[n]['MAE']  for n in names]
rmses = [results[n]['RMSE'] for n in names]
r2s   = [results[n]['R2']   for n in names]

colors = ['#4C72B0','#55A868','#C44E52','#8172B2']

axes[0].bar(names, maes, color=colors)
axes[0].set_title('Mean Absolute Error (EUR ) (lower is better)')
axes[0].set_ylabel('EUR ')
axes[0].tick_params(axis='x', rotation=20)
for i,v in enumerate(maes): axes[0].text(i, v+10, f'EUR {v:.0f}', ha='center', fontsize=9)

axes[1].bar(names, rmses, color=colors)
axes[1].set_title('RMSE (EUR ) (lower is better)')
axes[1].set_ylabel('EUR ')
axes[1].tick_params(axis='x', rotation=20)
for i,v in enumerate(rmses): axes[1].text(i, v+10, f'EUR {v:.0f}', ha='center', fontsize=9)

axes[2].bar(names, r2s, color=colors)
axes[2].set_title('R2 Score (higher is better)')
axes[2].set_ylabel('R2')
axes[2].set_ylim(0, 1.05)
axes[2].tick_params(axis='x', rotation=20)
for i,v in enumerate(r2s): axes[2].text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved outputs/model_comparison.png")

# ── 6. BEST MODEL DEEP-DIVE (Gradient Boosting) ──────────────
best_model = models['Gradient Boosting']
best_preds = results['Gradient Boosting']['preds']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Gradient Boosting - Detailed Analysis', fontsize=13, fontweight='bold')

axes[0].scatter(y_test, best_preds, alpha=0.3, s=15, color='steelblue')
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn,mx],[mn,mx], 'r--', linewidth=1.5, label='Perfect fit')
axes[0].set_xlabel('Actual Price/sqm (EUR )')
axes[0].set_ylabel('Predicted Price/sqm (EUR )')
axes[0].set_title(f'Actual vs Predicted  (R2={results["Gradient Boosting"]["R2"]:.4f})')
axes[0].legend()

importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
axes[1].barh(feat_imp.index, feat_imp.values, color='steelblue')
axes[1].set_title('Feature Importance')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('outputs/best_model_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved outputs/best_model_analysis.png")

# ── 7. RESIDUAL ANALYSIS ─────────────────────────────────────
residuals = y_test.values - best_preds
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Residual Analysis - Gradient Boosting', fontsize=13, fontweight='bold')

axes[0].scatter(best_preds, residuals, alpha=0.3, s=15, color='darkorange')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Predicted Price/sqm (EUR )')
axes[0].set_ylabel('Residual (EUR )')
axes[0].set_title('Residuals vs Predicted')

axes[1].hist(residuals, bins=40, color='darkorange', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Residual (EUR )')
axes[1].set_ylabel('Count')
axes[1].set_title('Residual Distribution')

plt.tight_layout()
plt.savefig('outputs/residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved outputs/residual_analysis.png")

# ── 8. CITY-LEVEL PREDICTION SUMMARY ─────────────────────────
print("\n[Results] City-level prediction accuracy:")
df_test = X_test.copy()
df_test['actual']    = y_test.values
df_test['predicted'] = best_preds
df_test['city'] = le.inverse_transform(df_test['city_encoded'])

city_metrics = df_test.groupby('city').apply(
    lambda g: pd.Series({
        'n': len(g),
        'actual_mean':    g['actual'].mean(),
        'predicted_mean': g['predicted'].mean(),
        'MAE':            mean_absolute_error(g['actual'], g['predicted']),
        'R2':             r2_score(g['actual'], g['predicted'])
    })
).round(2)
print(city_metrics.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(city_metrics))
w = 0.35
ax.bar(x - w/2, city_metrics['actual_mean'],    w, label='Actual Avg', color='steelblue')
ax.bar(x + w/2, city_metrics['predicted_mean'], w, label='Predicted Avg', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(city_metrics.index, rotation=20)
ax.set_ylabel('Avg Price/sqm (EUR )')
ax.set_title('Actual vs Predicted Average Price/sqm by City')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/city_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved outputs/city_predictions.png")

# ── 9. FINAL SUMMARY ─────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
best = results['Gradient Boosting']
print(f"Best Model      : Gradient Boosting Regressor")
print(f"R2 Score        : {best['R2']:.4f}")
print(f"MAE             : EUR {best['MAE']:,.2f} per sqm")
print(f"RMSE            : EUR {best['RMSE']:,.2f} per sqm")
print(f"\nAll plots saved to: outputs/")
print("=" * 60)

city_metrics.to_csv('outputs/city_prediction_results.csv')
print("  [OK] Saved outputs/city_prediction_results.csv")

df.to_csv('outputs/housing_dataset.csv', index=False)
print("  [OK] Saved outputs/housing_dataset.csv")
