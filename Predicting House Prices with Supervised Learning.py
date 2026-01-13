import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: GENERATE REALISTIC HOUSE DATA
# ============================================================================

def generate_house_data(n_houses=5000):
    """
    Generate realistic house price data with multiple features
    """
    np.random.seed(42)
    
    houses = []
    
    # Define neighborhoods with different characteristics
    neighborhoods = {
        'Downtown': {'base_price': 450000, 'price_per_sqft': 280, 'desirability': 1.3},
        'Suburban North': {'base_price': 380000, 'price_per_sqft': 220, 'desirability': 1.15},
        'Suburban South': {'base_price': 350000, 'price_per_sqft': 200, 'desirability': 1.1},
        'Suburban East': {'base_price': 320000, 'price_per_sqft': 190, 'desirability': 1.05},
        'Suburban West': {'base_price': 340000, 'price_per_sqft': 205, 'desirability': 1.08},
        'Urban Core': {'base_price': 500000, 'price_per_sqft': 320, 'desirability': 1.4},
        'Waterfront': {'base_price': 650000, 'price_per_sqft': 380, 'desirability': 1.6},
        'Historic District': {'base_price': 420000, 'price_per_sqft': 250, 'desirability': 1.25},
        'Industrial Area': {'base_price': 280000, 'price_per_sqft': 160, 'desirability': 0.9},
        'Rural': {'base_price': 250000, 'price_per_sqft': 140, 'desirability': 0.85}
    }
    
    for i in range(n_houses):
        # Basic features
        neighborhood = np.random.choice(list(neighborhoods.keys()))
        neighborhood_info = neighborhoods[neighborhood]
        
        bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.15, 0.35, 0.30, 0.10, 0.05])
        bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], 
                                     p=[0.10, 0.15, 0.30, 0.20, 0.15, 0.07, 0.03])
        
        # Square footage (correlated with bedrooms)
        base_sqft = 800 + bedrooms * 350
        square_feet = int(np.random.normal(base_sqft, 300))
        square_feet = max(500, min(square_feet, 6000))
        
        # Lot size
        lot_size = int(np.random.normal(5000 + square_feet * 1.5, 2000))
        lot_size = max(1000, lot_size)
        
        # Age of house
        year_built = np.random.randint(1950, 2024)
        house_age = 2024 - year_built
        
        # Additional features
        has_garage = np.random.choice([0, 1], p=[0.20, 0.80])
        garage_spaces = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.4, 0.1]) if has_garage else 0
        
        has_pool = np.random.choice([0, 1], p=[0.80, 0.20])
        has_basement = np.random.choice([0, 1], p=[0.60, 0.40])
        basement_finished = np.random.choice([0, 1], p=[0.50, 0.50]) if has_basement else 0
        
        floors = np.random.choice([1, 2, 3], p=[0.30, 0.60, 0.10])
        
        condition = np.random.choice(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
                                    p=[0.05, 0.15, 0.45, 0.25, 0.10])
        condition_score = {'Poor': 0.8, 'Fair': 0.9, 'Good': 1.0, 'Very Good': 1.1, 'Excellent': 1.25}
        
        # Proximity features (in miles)
        distance_to_downtown = np.random.uniform(0.5, 25)
        distance_to_school = np.random.uniform(0.2, 5)
        distance_to_transit = np.random.uniform(0.1, 3)
        
        # School rating (1-10)
        school_rating = np.random.randint(3, 11)
        
        # Crime rate index (lower is better, 1-10)
        crime_rate = np.random.randint(1, 11)
        
        # Renovation status
        recently_renovated = np.random.choice([0, 1], p=[0.75, 0.25])
        
        # View quality
        view_quality = np.random.choice(['None', 'Fair', 'Good', 'Excellent'],
                                       p=[0.60, 0.20, 0.15, 0.05])
        view_score = {'None': 1.0, 'Fair': 1.05, 'Good': 1.12, 'Excellent': 1.25}
        
        # HOA fees (monthly)
        has_hoa = np.random.choice([0, 1], p=[0.60, 0.40])
        hoa_fee = int(np.random.normal(150, 50)) if has_hoa else 0
        hoa_fee = max(0, hoa_fee)
        
        # Property tax rate (percentage)
        property_tax_rate = np.random.uniform(0.8, 2.5)
        
        # Calculate price based on all features
        base_price = neighborhood_info['base_price']
        price_per_sqft = neighborhood_info['price_per_sqft']
        desirability = neighborhood_info['desirability']
        
        # Price calculation with various factors
        price = base_price + (square_feet * price_per_sqft)
        
        # Adjustments
        price *= condition_score[condition]
        price *= view_score[view_quality]
        price *= desirability
        
        # Age depreciation (newer houses worth more)
        age_factor = 1 - (house_age * 0.005)  # 0.5% per year
        age_factor = max(0.7, age_factor)  # minimum 70% of base value
        price *= age_factor
        
        # Features that add value
        if has_pool:
            price += 35000
        if has_basement:
            price += 25000 * (1.5 if basement_finished else 1)
        if garage_spaces > 0:
            price += garage_spaces * 15000
        if recently_renovated:
            price *= 1.15
        
        # Location factors
        price *= (1 - distance_to_downtown * 0.008)  # Further from downtown = cheaper
        price *= (1 + (school_rating / 100))  # Better schools = more expensive
        price *= (1 - (crime_rate / 200))  # Higher crime = cheaper
        price *= (1 - (distance_to_transit * 0.02))  # Closer to transit = more expensive
        
        # Add some random noise
        price *= np.random.normal(1.0, 0.08)
        
        # Ensure positive price
        price = max(50000, int(price))
        
        houses.append({
            'house_id': f'H{i+1:05d}',
            'price': price,
            'neighborhood': neighborhood,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'square_feet': square_feet,
            'lot_size': lot_size,
            'year_built': year_built,
            'house_age': house_age,
            'floors': floors,
            'condition': condition,
            'has_garage': has_garage,
            'garage_spaces': garage_spaces,
            'has_pool': has_pool,
            'has_basement': has_basement,
            'basement_finished': basement_finished,
            'view_quality': view_quality,
            'recently_renovated': recently_renovated,
            'distance_to_downtown': round(distance_to_downtown, 2),
            'distance_to_school': round(distance_to_school, 2),
            'distance_to_transit': round(distance_to_transit, 2),
            'school_rating': school_rating,
            'crime_rate': crime_rate,
            'has_hoa': has_hoa,
            'hoa_fee_monthly': hoa_fee,
            'property_tax_rate': round(property_tax_rate, 2)
        })
    
    return pd.DataFrame(houses)

# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """
    Preprocess the house data for machine learning
    """
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    df_processed = df.copy()
    
    # Check for missing values
    print(f"\nMissing values:\n{df_processed.isnull().sum()}")
    
    # Create additional engineered features
    df_processed['price_per_sqft'] = df_processed['price'] / df_processed['square_feet']
    df_processed['bed_bath_ratio'] = df_processed['bedrooms'] / df_processed['bathrooms']
    df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['bathrooms']
    df_processed['is_new_construction'] = (df_processed['house_age'] < 5).astype(int)
    df_processed['is_old_house'] = (df_processed['house_age'] > 50).astype(int)
    df_processed['has_any_parking'] = (df_processed['garage_spaces'] > 0).astype(int)
    df_processed['luxury_score'] = (
        (df_processed['has_pool'] * 2) +
        (df_processed['basement_finished'] * 1.5) +
        (df_processed['garage_spaces'] * 0.5) +
        (df_processed['bedrooms'] > 4).astype(int) * 1
    )
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['neighborhood', 'condition', 'view_quality']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    print("\nCategorical encoding completed:")
    for col in categorical_cols:
        print(f"  {col}: {df_processed[col].nunique()} unique values")
    
    # Create feature statistics
    print("\n" + "-" * 80)
    print("FEATURE STATISTICS")
    print("-" * 80)
    
    numeric_features = ['price', 'square_feet', 'lot_size', 'bedrooms', 
                       'bathrooms', 'house_age', 'school_rating']
    print(df_processed[numeric_features].describe().round(2))
    
    return df_processed, label_encoders

# ============================================================================
# PART 3: TRAIN MULTIPLE REGRESSION MODELS
# ============================================================================

def train_models(df_processed):
    """
    Train multiple regression models and compare performance
    """
    print("\n" + "=" * 80)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    # Select features for modeling
    feature_cols = [
        'bedrooms', 'bathrooms', 'square_feet', 'lot_size', 'house_age',
        'floors', 'garage_spaces', 'has_pool', 'has_basement', 'basement_finished',
        'recently_renovated', 'distance_to_downtown', 'distance_to_school',
        'distance_to_transit', 'school_rating', 'crime_rate', 'has_hoa',
        'hoa_fee_monthly', 'property_tax_rate', 'neighborhood_encoded',
        'condition_encoded', 'view_quality_encoded', 'bed_bath_ratio',
        'total_rooms', 'is_new_construction', 'is_old_house', 'luxury_score'
    ]
    
    X = df_processed[feature_cols]
    y = df_processed['price']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1000),
        'Decision Tree': DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, 
                                               min_samples_split=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=7,
                                                       learning_rate=0.1, random_state=42)
    }
    
    # Train and evaluate each model
    results = []
    trained_models = {}
    
    print("\n" + "-" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    print(f"{'Model':<25} {'MAE':<15} {'RMSE':<15} {'R² Score':<12} {'CV Score':<12}")
    print("-" * 80)
    
    for name, model in models.items():
        # Train model
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            X_cv = X_train_scaled
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            X_cv = X_train
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_cv, y_train, cv=5, 
                                    scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_R2': cv_mean
        })
        
        trained_models[name] = model
        
        print(f"{name:<25} ${mae:>12,.0f}  ${rmse:>12,.0f}  {r2:>10.4f}  {cv_mean:>10.4f}")
    
    results_df = pd.DataFrame(results)
    best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
    best_model = trained_models[best_model_name]
    
    print("-" * 80)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   R² Score: {results_df.loc[results_df['R2'].idxmax(), 'R2']:.4f}")
    
    return trained_models, scaler, feature_cols, X_test, y_test, best_model_name

# ============================================================================
# PART 4: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(trained_models, feature_cols, best_model_name):
    """
    Analyze which features are most important for prediction
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Get feature importance from tree-based models
    if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
        model = trained_models[best_model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop 15 Most Important Features ({best_model_name}):")
            print("-" * 80)
            for idx, row in importance_df.head(15).iterrows():
                bar = '█' * int(row['Importance'] * 100)
                print(f"{row['Feature']:<30} {row['Importance']:>6.4f} {bar}")
            
            return importance_df
    
    return None

# ============================================================================
# PART 5: MAKE PREDICTIONS ON NEW DATA
# ============================================================================

def predict_house_prices(trained_models, scaler, feature_cols, best_model_name):
    """
    Make predictions on new house data
    """
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS ON NEW HOUSES")
    print("=" * 80)
    
    # Create sample houses for prediction
    new_houses = [
        {
            'bedrooms': 3, 'bathrooms': 2, 'square_feet': 1800, 'lot_size': 5000,
            'house_age': 10, 'floors': 2, 'garage_spaces': 2, 'has_pool': 0,
            'has_basement': 1, 'basement_finished': 1, 'recently_renovated': 1,
            'distance_to_downtown': 8.5, 'distance_to_school': 1.2,
            'distance_to_transit': 0.8, 'school_rating': 8, 'crime_rate': 3,
            'has_hoa': 1, 'hoa_fee_monthly': 150, 'property_tax_rate': 1.2,
            'neighborhood_encoded': 2, 'condition_encoded': 3, 'view_quality_encoded': 2,
            'description': '3BR/2BA Suburban Home with Finished Basement'
        },
        {
            'bedrooms': 5, 'bathrooms': 3.5, 'square_feet': 3500, 'lot_size': 8000,
            'house_age': 2, 'floors': 2, 'garage_spaces': 3, 'has_pool': 1,
            'has_basement': 1, 'basement_finished': 1, 'recently_renovated': 0,
            'distance_to_downtown': 3.2, 'distance_to_school': 0.5,
            'distance_to_transit': 0.3, 'school_rating': 10, 'crime_rate': 2,
            'has_hoa': 1, 'hoa_fee_monthly': 300, 'property_tax_rate': 1.8,
            'neighborhood_encoded': 6, 'condition_encoded': 4, 'view_quality_encoded': 3,
            'description': '5BR/3.5BA Luxury Waterfront Home'
        },
        {
            'bedrooms': 2, 'bathrooms': 1, 'square_feet': 950, 'lot_size': 2500,
            'house_age': 45, 'floors': 1, 'garage_spaces': 1, 'has_pool': 0,
            'has_basement': 0, 'basement_finished': 0, 'recently_renovated': 0,
            'distance_to_downtown': 15.0, 'distance_to_school': 3.0,
            'distance_to_transit': 2.5, 'school_rating': 5, 'crime_rate': 7,
            'has_hoa': 0, 'hoa_fee_monthly': 0, 'property_tax_rate': 1.0,
            'neighborhood_encoded': 8, 'condition_encoded': 1, 'view_quality_encoded': 0,
            'description': '2BR/1BA Starter Home'
        },
        {
            'bedrooms': 4, 'bathrooms': 3, 'square_feet': 2400, 'lot_size': 6000,
            'house_age': 5, 'floors': 2, 'garage_spaces': 2, 'has_pool': 0,
            'has_basement': 1, 'basement_finished': 0, 'recently_renovated': 1,
            'distance_to_downtown': 1.5, 'distance_to_school': 0.8,
            'distance_to_transit': 0.2, 'school_rating': 9, 'crime_rate': 2,
            'has_hoa': 1, 'hoa_fee_monthly': 250, 'property_tax_rate': 1.5,
            'neighborhood_encoded': 0, 'condition_encoded': 4, 'view_quality_encoded': 2,
            'description': '4BR/3BA Modern Downtown Home'
        }
    ]
    
    predictions_list = []
    
    for house in new_houses:
        description = house.pop('description')
        
        # Engineer features
        house_df = pd.DataFrame([house])
        house_df['bed_bath_ratio'] = house_df['bedrooms'] / house_df['bathrooms']
        house_df['total_rooms'] = house_df['bedrooms'] + house_df['bathrooms']
        house_df['is_new_construction'] = (house_df['house_age'] < 5).astype(int)
        house_df['is_old_house'] = (house_df['house_age'] > 50).astype(int)
        house_df['has_any_parking'] = (house_df['garage_spaces'] > 0).astype(int)
        house_df['luxury_score'] = (
            (house_df['has_pool'] * 2) +
            (house_df['basement_finished'] * 1.5) +
            (house_df['garage_spaces'] * 0.5) +
            (house_df['bedrooms'] > 4).astype(int) * 1
        )
        
        # Ensure all features are present
        X_new = house_df[feature_cols]
        
        # Make prediction with best model
        best_model = trained_models[best_model_name]
        
        if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            X_new_scaled = scaler.transform(X_new)
            predicted_price = best_model.predict(X_new_scaled)[0]
        else:
            predicted_price = best_model.predict(X_new)[0]
        
        predictions_list.append({
            'Description': description,
            'Predicted Price': f"${predicted_price:,.0f}"
        })
        
        print(f"\n{description}")
        print(f"  Predicted Price: ${predicted_price:,.0f}")
    
    return predictions_list

# ============================================================================
# PART 6: MODEL INSIGHTS AND RECOMMENDATIONS
# ============================================================================

def generate_insights(df, results_df):
    """
    Generate insights and recommendations based on the data
    """
    print("\n" + "=" * 80)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. PRICE DISTRIBUTION BY NEIGHBORHOOD")
    print("-" * 80)
    neighborhood_stats = df.groupby('neighborhood')['price'].agg(['mean', 'median', 'std']).round(0)
    neighborhood_stats = neighborhood_stats.sort_values('mean', ascending=False)
    print(neighborhood_stats.to_string())
    
    print("\n\n2. IMPACT OF KEY FEATURES ON PRICE")
    print("-" * 80)
    
    # Pool impact
    pool_impact = df.groupby('has_pool')['price'].mean()
    pool_premium = ((pool_impact[1] / pool_impact[0]) - 1) * 100
    print(f"Houses with Pool: +{pool_premium:.1f}% price premium")
    
    # Garage impact
    garage_stats = df.groupby('garage_spaces')['price'].mean()
    print(f"Average price increase per garage space: ${(garage_stats[1] - garage_stats[0]):,.0f}")
    
    # Renovation impact
    reno_impact = df.groupby('recently_renovated')['price'].mean()
    reno_premium = ((reno_impact[1] / reno_impact[0]) - 1) * 100
    print(f"Recently Renovated: +{reno_premium:.1f}% price premium")
    
    # School rating correlation
    school_corr = df[['school_rating', 'price']].corr().iloc[0, 1]
    print(f"School Rating Correlation: {school_corr:.3f}")
    
    print("\n\n3. OPTIMAL HOUSE CHARACTERISTICS")
    print("-" * 80)
    top_10_percent = df.nlargest(int(len(df) * 0.1), 'price')
    
    print(f"Average characteristics of top 10% most expensive houses:")
    print(f"  Bedrooms: {top_10_percent['bedrooms'].mean():.1f}")
    print(f"  Bathrooms: {top_10_percent['bathrooms'].mean():.1f}")
    print(f"  Square Feet: {top_10_percent['square_feet'].mean():,.0f}")
    print(f"  Age: {top_10_percent['house_age'].mean():.1f} years")
    print(f"  Pool: {top_10_percent['has_pool'].mean()*100:.0f}%")
    print(f"  Garage Spaces: {top_10_percent['garage_spaces'].mean():.1f}")
    
    print("\n\n4. MODEL RECOMMENDATIONS")
    print("-" * 80)
    print("✓ Use the trained model for automated property valuation")
    print("✓ Location (neighborhood) is the strongest price predictor")
    print("✓ Square footage and lot size significantly impact value")
    print("✓ School ratings and crime rates affect desirability")
    print("✓ Recent renovations can add 15%+ to property value")
    print("✓ Pool and finished basement are premium features")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HOUSE PRICE PREDICTION SYSTEM")
    print("=" * 80)
    
    # Step 1: Generate data
    print("\n[1/6] Generating house price data...")
    df = generate_house_data(n_houses=5000)
    print(f"✓ Generated {len(df)} house records")
    
    print("\nSample of generated data:")
    display_cols = ['price', 'neighborhood', 'bedrooms', 'bathrooms', 
                    'square_feet', 'house_age', 'condition']
    print(df[display_cols].head(10))
    
    # Step 2: Preprocess data
    print("\n\n[2/6] Preprocessing data...")
    df_processed, label_encoders = preprocess_data(df)
    print("✓ Data preprocessing completed")
    
    # Step 3: Train models
    print("\n\n[3/6] Training regression models...")
    trained_models, scaler, feature_cols, X_test, y_test, best_model_name = train_models(df_processed)
    print("✓ Model training completed")
    
    # Step 4: Feature importance
    print("\n\n[4/6] Analyzing feature importance...")
    importance_df = analyze_feature_importance(trained_models, feature_cols, best_model_name)
    print("✓ Feature importance analysis completed")
    
    # Step 5: Make predictions
    print("\n\n[5/6] Making predictions on sample houses...")
    predictions = predict_house_prices(trained_models, scaler, feature_cols, best_model_name)
    print("✓ Predictions completed")
    
    # Step 6: Generate insights
    print("\n\n[6/6] Generating insights...")
    results_df = pd.DataFrame()  # Placeholder for results
    generate_insights(df, results_df)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SYSTEM READY")
    print("=" * 80)
    print(f"\n✓ Dataset: {len(df)} houses across {df['neighborhood'].nunique()} neighborhoods")
    print(f"✓ Best Model: {best_model_name}")
    print(f"✓ Features Used: {len(feature_cols)}")
    print(f"✓ Price Range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"✓ Average Price: ${df['price'].mean():,.0f}")
    print("\n💡 The model is now ready to predict house prices based on features!")
    print("   You can use this system for property valuation and market analysis.")