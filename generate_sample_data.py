import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

print("🔄 Generating sample data for Dynamic Pricing Strategy...")
print("Using exact data structure specifications...")

# Create data directory if it doesn't exist
os.makedirs('D:/Narayan/Work/hackathon-dynamic-pricing/data/sample', exist_ok=True)
base_path = 'D:/Narayan/Work/hackathon-dynamic-pricing/data/sample'

# Generate date range for the last 6 months
end_date = datetime.now()
start_date = end_date - timedelta(days=180)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"📊 Total days: {len(date_range)}")

# 1. SALES_DATA.CSV
print("\n1️⃣ Generating sales_data.csv...")
sales_data = []

for date in date_range:
    # Generate 5-25 transactions per day (varying by day type)
    day_of_week = date.weekday()
    if day_of_week >= 5:  # Weekend
        daily_transactions = np.random.randint(8, 25)
    else:  # Weekday
        daily_transactions = np.random.randint(5, 18)
    
    # Add seasonal effects
    month = date.month
    seasonal_multiplier = 1.0
    if month in [11, 12]:  # Holiday season
        seasonal_multiplier = 1.4
    elif month in [6, 7, 8]:  # Summer
        seasonal_multiplier = 1.2
    elif month in [1, 2]:  # Post-holiday slump
        seasonal_multiplier = 0.8
    
    daily_transactions = int(daily_transactions * seasonal_multiplier)
    
    for _ in range(daily_transactions):
        # Base MRP with some variation
        mrp = np.random.uniform(95.0, 125.0)
        
        # NoPromoPrice is typically 5-15% less than MRP
        no_promo_discount = np.random.uniform(0.05, 0.15)
        no_promo_price = mrp * (1 - no_promo_discount)
        
        # SellingPrice can be same as NoPromoPrice or have additional promotional discount
        additional_discount = np.random.choice([0, 0.05, 0.10, 0.15], p=[0.4, 0.3, 0.2, 0.1])
        selling_price = no_promo_price * (1 - additional_discount)
        
        # Units sold depends on discount level and day type
        base_units = 45
        price_elasticity = (mrp - selling_price) / mrp  # Higher discount = more units
        units_multiplier = 1 + (price_elasticity * 2)  # Price elasticity effect
        
        if day_of_week >= 5:  # Weekend boost
            units_multiplier *= 1.3
        
        units_sold = max(1, int(np.random.poisson(base_units * units_multiplier * seasonal_multiplier)))
        
        sales_data.append({
            'TransactionDate': date.strftime('%Y-%m-%d'),
            'MRP': round(mrp, 2),
            'NoPromoPrice': round(no_promo_price, 2),
            'SellingPrice': round(selling_price, 2),
            'UnitsSold': units_sold
        })

sales_df = pd.DataFrame(sales_data)
sales_df.to_csv(f'{base_path}/sales_data.csv', index=False)
print(f"✅ Generated {len(sales_df):,} sales transaction records")
print(f"   Average daily transactions: {len(sales_df)/len(date_range):.1f}")

# 2. COMPETITOR_DATA.CSV
print("\n2️⃣ Generating competitor_data.csv...")
competitor_data = []

# Define competitor brands with their positioning
competitors = [
    {'Brand': 'ArielPro', 'positioning': 'premium'},
    {'Brand': 'SurfExcel', 'positioning': 'premium'},
    {'Brand': 'RinAdvanced', 'positioning': 'mid-range'},
    {'Brand': 'WheelActive', 'positioning': 'budget'},
    {'Brand': 'GhariBrand', 'positioning': 'budget'}
]

for date in date_range:
    for competitor in competitors:
        brand = competitor['Brand']
        positioning = competitor['positioning']
        
        # Base MRP varies by positioning
        if positioning == 'premium':
            base_mrp = np.random.uniform(110.0, 140.0)
        elif positioning == 'mid-range':
            base_mrp = np.random.uniform(85.0, 115.0)
        else:  # budget
            base_mrp = np.random.uniform(65.0, 95.0)
        
        # Discount rate varies by brand strategy and day
        day_of_week = date.weekday()
        if day_of_week >= 5:  # Weekend promotions
            discount_rate = np.random.uniform(8.0, 25.0)
        else:
            discount_rate = np.random.uniform(3.0, 18.0)
        
        # Premium brands generally have lower discount rates
        if positioning == 'premium':
            discount_rate *= 0.7
        elif positioning == 'budget':
            discount_rate *= 1.2
        
        # Calculate derived prices
        base_price = base_mrp * (1 - discount_rate/100)
        
        # Final price might have additional small adjustments
        final_price_adjustment = np.random.uniform(-0.02, 0.02)
        final_price = base_price * (1 + final_price_adjustment)
        
        competitor_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Brand': brand,
            'MRP': round(base_mrp, 2),
            'DiscountRate': round(discount_rate, 2),
            'BasePrice': round(base_price, 2),
            'FinalPrice': round(final_price, 2)
        })

competitor_df = pd.DataFrame(competitor_data)
competitor_df.to_csv(f'{base_path}/competitor_data.csv', index=False)
print(f"✅ Generated {len(competitor_df):,} competitor pricing records")
print(f"   Brands tracked: {len(competitors)}")

# 3. CUSTOMER_BEHAVIOR_DATA.CSV
print("\n3️⃣ Generating customer_behavior_data.csv...")
customer_data = []

for i, date in enumerate(date_range):
    # Add long-term trends (improving metrics over time)
    trend_factor = 1 + (i / len(date_range)) * 0.15  # 15% improvement over 6 months
    
    # Day-of-week effects
    day_of_week = date.weekday()
    if day_of_week >= 5:  # Weekend
        weekend_ctr_boost = 0.85  # Lower CTR on weekends
        weekend_session_boost = 1.3  # Longer sessions
    else:
        weekend_ctr_boost = 1.0
        weekend_session_boost = 1.0
    
    # Seasonal effects on behavior
    month = date.month
    seasonal_engagement = 1.0
    if month in [11, 12]:  # Holiday season - higher engagement
        seasonal_engagement = 1.2
    elif month in [1, 2]:  # Post-holiday - lower engagement
        seasonal_engagement = 0.9
    
    # Generate metrics with realistic correlations
    base_ctr = 0.028 * trend_factor * weekend_ctr_boost * seasonal_engagement
    ctr = max(0.005, base_ctr + np.random.normal(0, 0.008))
    
    # Higher CTR typically correlates with lower bounce rate
    bounce_rate_base = 0.35 - (ctr - 0.025) * 2  # Inverse correlation
    bounce_rate = max(0.15, min(0.65, bounce_rate_base + np.random.normal(0, 0.08)))
    
    # Abandoned cart rate is influenced by seasonal factors
    abandoned_cart_base = 0.22
    if month in [11, 12]:  # Holiday urgency reduces abandonment
        abandoned_cart_base *= 0.85
    abandoned_cart_rate = max(0.10, abandoned_cart_base + np.random.normal(0, 0.06))
    
    # Funnel drops are correlated with bounce rate
    funnel_view_to_cart = max(0.08, bounce_rate * 0.6 + np.random.normal(0, 0.04))
    funnel_cart_to_checkout = max(0.05, abandoned_cart_rate * 0.8 + np.random.normal(0, 0.03))
    
    # Returning visitor ratio improves over time and is higher on weekends
    returning_visitor_base = 0.35 * trend_factor * (1.2 if day_of_week >= 5 else 1.0)
    returning_visitor_ratio = max(0.15, min(0.75, returning_visitor_base + np.random.normal(0, 0.08)))
    
    # Session duration is longer on weekends and during holidays
    session_duration_base = 145 * weekend_session_boost * seasonal_engagement
    avg_session_duration = max(45, session_duration_base + np.random.normal(0, 35))
    
    customer_data.append({
        'Date': date.strftime('%Y-%m-%d'),
        'CTR': round(ctr, 4),
        'AbandonedCartRate': round(abandoned_cart_rate, 3),
        'BounceRate': round(bounce_rate, 3),
        'FunnelDrop_ViewToCart': round(funnel_view_to_cart, 3),
        'FunnelDrop_CartToCheckout': round(funnel_cart_to_checkout, 3),
        'ReturningVisitorRatio': round(returning_visitor_ratio, 3),
        'AvgSessionDuration_sec': round(avg_session_duration, 1)
    })

customer_df = pd.DataFrame(customer_data)
customer_df.to_csv(f'{base_path}/customer_behavior_data.csv', index=False)
print(f"✅ Generated {len(customer_df):,} customer behavior records")

# 4. INVENTORY_DATA.CSV
print("\n4️⃣ Generating inventory_data.csv...")
inventory_data = []

# Define fulfillment centers
fulfillment_centers = [
    {'FC_ID': 'FC_MUM_01', 'IsMetro': True, 'capacity_multiplier': 1.4},
    {'FC_ID': 'FC_DEL_01', 'IsMetro': True, 'capacity_multiplier': 1.5},
    {'FC_ID': 'FC_BLR_01', 'IsMetro': True, 'capacity_multiplier': 1.3},
    {'FC_ID': 'FC_HYD_01', 'IsMetro': True, 'capacity_multiplier': 1.1},
    {'FC_ID': 'FC_CHN_01', 'IsMetro': True, 'capacity_multiplier': 1.2},
    {'FC_ID': 'FC_JAI_01', 'IsMetro': False, 'capacity_multiplier': 0.7},
    {'FC_ID': 'FC_LKO_01', 'IsMetro': False, 'capacity_multiplier': 0.6},
    {'FC_ID': 'FC_IND_01', 'IsMetro': False, 'capacity_multiplier': 0.8},
    {'FC_ID': 'FC_NAG_01', 'IsMetro': False, 'capacity_multiplier': 0.5}
]

# Track stock levels over time for each FC
fc_stock_tracking = {}

for date in date_range:
    # Calculate day index for stock continuity
    day_index = (date - start_date).days
    
    for fc in fulfillment_centers:
        fc_id = fc['FC_ID']
        is_metro = fc['IsMetro']
        capacity_mult = fc['capacity_multiplier']
        
        # Initialize stock for first day
        if fc_id not in fc_stock_tracking:
            fc_stock_tracking[fc_id] = {
                'current_stock': int(np.random.randint(300, 800) * capacity_mult),
                'reorder_point': int(np.random.randint(120, 200) * capacity_mult),
                'safety_stock': int(np.random.randint(80, 120) * capacity_mult)
            }
        
        current_stock = fc_stock_tracking[fc_id]['current_stock']
        reorder_point = fc_stock_tracking[fc_id]['reorder_point']
        safety_stock = fc_stock_tracking[fc_id]['safety_stock']
        
        # Generate demand based on metro status and seasonal factors
        base_demand = 150 if is_metro else 80
        base_demand = int(base_demand * capacity_mult)
        
        # Seasonal demand adjustments
        month = date.month
        day_of_week = date.weekday()
        
        seasonal_factor = 1.0
        if month in [11, 12]:  # Holiday season
            seasonal_factor = 1.5
        elif month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.2
        elif month in [1, 2]:  # Post-holiday
            seasonal_factor = 0.8
        
        weekend_factor = 1.3 if day_of_week >= 5 else 1.0
        
        # Generate demand with some randomness
        demand = max(10, int(np.random.poisson(base_demand * seasonal_factor * weekend_factor)))
        
        # Calculate fulfillment
        demand_fulfilled = min(demand, current_stock)
        backorders = max(0, demand - demand_fulfilled)
        stock_end = max(0, current_stock - demand_fulfilled)
        
        # Reorder logic
        order_placed = 1 if stock_end <= reorder_point else 0
        
        if order_placed:
            # Order quantity to bring stock back to max capacity
            max_capacity = int(1000 * capacity_mult)
            order_qty = max_capacity - stock_end
        else:
            order_qty = 0
        
        # Lead time varies by location (metro vs non-metro)
        if is_metro:
            lead_time = np.random.uniform(1.5, 4.0)
        else:
            lead_time = np.random.uniform(2.5, 6.5)
        
        # Add previous day's orders to current stock (simplified)
        if day_index > 0:
            # Simulate receiving orders placed 3-5 days ago
            if np.random.random() < 0.2:  # 20% chance of receiving stock
                stock_replenishment = np.random.randint(200, 600) * capacity_mult
                stock_end += int(stock_replenishment)
        
        # Update tracking for next day
        fc_stock_tracking[fc_id]['current_stock'] = stock_end
        
        inventory_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'FC_ID': fc_id,
            'IsMetro': is_metro,
            'StockStart': current_stock,
            'Demand': demand,
            'DemandFulfilled': demand_fulfilled,
            'Backorders': backorders,
            'StockEnd': stock_end,
            'ReorderPoint': reorder_point,
            'OrderPlaced': order_placed,
            'OrderQty': order_qty,
            'LeadTimeFloat': round(lead_time, 1),
            'SafetyStock': safety_stock
        })

inventory_df = pd.DataFrame(inventory_data)
inventory_df.to_csv(f'{base_path}/inventory_data.csv', index=False)
print(f"✅ Generated {len(inventory_df):,} inventory records")
print(f"   Fulfillment centers: {len(fulfillment_centers)}")

# GENERATE SUMMARY STATISTICS
print(f"\n📈 DATA GENERATION SUMMARY")
print("=" * 50)

print(f"\n1️⃣ SALES DATA:")
print(f"   • Records: {len(sales_df):,}")
print(f"   • Date range: {sales_df['TransactionDate'].min()} to {sales_df['TransactionDate'].max()}")
print(f"   • Total units sold: {sales_df['UnitsSold'].sum():,}")
print(f"   • Avg units per transaction: {sales_df['UnitsSold'].mean():.1f}")
print(f"   • Price range: ₹{sales_df['SellingPrice'].min():.2f} - ₹{sales_df['SellingPrice'].max():.2f}")

print(f"\n2️⃣ COMPETITOR DATA:")
print(f"   • Records: {len(competitor_df):,}")
print(f"   • Brands: {competitor_df['Brand'].nunique()}")
print(f"   • Avg discount rate: {competitor_df['DiscountRate'].mean():.1f}%")
print(f"   • Price range: ₹{competitor_df['FinalPrice'].min():.2f} - ₹{competitor_df['FinalPrice'].max():.2f}")

print(f"\n3️⃣ CUSTOMER BEHAVIOR DATA:")
print(f"   • Records: {len(customer_df):,}")
print(f"   • Avg CTR: {customer_df['CTR'].mean():.3f}")
print(f"   • Avg cart abandonment: {customer_df['AbandonedCartRate'].mean():.1%}")
print(f"   • Avg session duration: {customer_df['AvgSessionDuration_sec'].mean():.0f} seconds")

print(f"\n4️⃣ INVENTORY DATA:")
print(f"   • Records: {len(inventory_df):,}")
print(f"   • Fulfillment centers: {inventory_df['FC_ID'].nunique()}")
print(f"   • Metro centers: {inventory_df[inventory_df['IsMetro']==True]['FC_ID'].nunique()}")
print(f"   • Avg daily demand: {inventory_df['Demand'].mean():.0f}")
print(f"   • Avg fulfillment rate: {(inventory_df['DemandFulfilled'].sum()/inventory_df['Demand'].sum()):.1%}")

print(f"\n💾 FILES SAVED TO:")
print(f"   📁 {base_path}/")
print(f"   📄 sales_data.csv")
print(f"   📄 competitor_data.csv") 
print(f"   📄 customer_behavior_data.csv")
print(f"   📄 inventory_data.csv")

print(f"\n✅ DATA GENERATION COMPLETED SUCCESSFULLY!")
