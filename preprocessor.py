import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def process_demolition_data(csv_path='ma_structures_with_demolition_FINAL.csv'):
    """
    Process MA structures data focusing on Boston demolitions and output JSON for dashboard
    Filters for buildings built in 1940 or later
    """

    print("Loading data...")
    # Load the data
    df = pd.read_csv(csv_path, low_memory=False)

    # Filter for Boston demolitions only
    print("Filtering for Boston demolitions...")
    # Filter for records with demolition data
    demo_df = df[df['DEMOLITION_TYPE'].notna()].copy()

    # Filter for Boston area (using PROP_CITY)
    boston_cities = ['BOSTON', 'CAMBRIDGE', 'SOMERVILLE', 'BROOKLINE', 'QUINCY',
                     'NEWTON', 'WATERTOWN', 'CHELSEA', 'REVERE', 'EVERETT']
    demo_df = demo_df[demo_df['PROP_CITY'].str.upper().isin(boston_cities)]

    print(f"Found {len(demo_df)} demolition records in Boston area")

    # ========== NEW FILTER: Buildings built in 1940 or later ==========
    print("Filtering for buildings built in 1940 or later...")
    pre_1940_count = len(demo_df)
    demo_df = demo_df[demo_df['year_built'] >= 1940].copy()
    post_1940_count = len(demo_df)
    print(f"Removed {pre_1940_count - post_1940_count} pre-1940 buildings")
    print(f"Remaining records: {post_1940_count}")
    # ================================================================

    # Convert dates
    demo_df['DEMOLITION_DATE'] = pd.to_datetime(demo_df['DEMOLITION_DATE'], errors='coerce')
    demo_df['demolition_year'] = demo_df['DEMOLITION_DATE'].dt.year

    # Calculate lifespan
    demo_df['lifespan'] = demo_df['demolition_year'] - demo_df['year_built']

    # Clean data - remove invalid lifespans
    demo_df = demo_df[(demo_df['lifespan'] > 0) & (demo_df['lifespan'] < 500)]

    # Remove duplicates based on BUILD_ID, keeping the most recent demolition
    demo_df = demo_df.sort_values('DEMOLITION_DATE').drop_duplicates('BUILD_ID', keep='last')

    print(f"After cleaning: {len(demo_df)} unique buildings")

    # Initialize result dictionary
    result = {}

    # 1. Summary Statistics
    print("Calculating summary statistics...")
    result['summary_stats'] = {
        'total_demolitions': int(len(demo_df)),
        'average_lifespan': float(demo_df['lifespan'].mean()),
        'median_lifespan': float(demo_df['lifespan'].median()),
        'extdem_count': int((demo_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
        'intdem_count': int((demo_df['DEMOLITION_TYPE'] == 'INTDEM').sum()),
        'raze_count': int((demo_df['DEMOLITION_TYPE'] == 'RAZE').sum()),
        'min_year_built': int(demo_df['year_built'].min()),  # Add min year built
        'max_year_built': int(demo_df['year_built'].max())  # Add max year built
    }

    # NEW: Calculate average lifespan by demolition type
    print("Calculating average lifespan by demolition type...")
    material_lifespan_demo_avg = {}
    for demo_type in ['RAZE', 'EXTDEM', 'INTDEM']:
        demo_type_df = demo_df[demo_df['DEMOLITION_TYPE'] == demo_type]
        if len(demo_type_df) > 0:
            material_lifespan_demo_avg[demo_type] = float(demo_type_df['lifespan'].mean())
        else:
            material_lifespan_demo_avg[demo_type] = 0

    result['material_lifespan_demo_avg'] = material_lifespan_demo_avg

    # 2. Yearly Stacked Data
    print("Processing yearly demolition data...")
    yearly_data = []
    years = sorted(demo_df['demolition_year'].dropna().unique())

    for year in years:
        year_df = demo_df[demo_df['demolition_year'] == year]
        yearly_data.append({
            'year': int(year),
            'EXTDEM': int((year_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
            'INTDEM': int((year_df['DEMOLITION_TYPE'] == 'INTDEM').sum()),
            'RAZE': int((year_df['DEMOLITION_TYPE'] == 'RAZE').sum())
        })

    result['yearly_stacked'] = yearly_data

    # 3. Lifespan Distribution (10-year bins)
    print("Creating lifespan distribution...")
    max_lifespan = int(demo_df['lifespan'].max())
    bin_size_10 = 10
    lifespan_dist_10 = []

    for i in range(0, min(max_lifespan + bin_size_10, 300), bin_size_10):
        bin_df = demo_df[(demo_df['lifespan'] >= i) & (demo_df['lifespan'] < i + bin_size_10)]
        if len(bin_df) > 0:
            lifespan_dist_10.append({
                'range': f"{i}-{i + bin_size_10}",
                'EXTDEM': int((bin_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
                'INTDEM': int((bin_df['DEMOLITION_TYPE'] == 'INTDEM').sum()),
                'RAZE': int((bin_df['DEMOLITION_TYPE'] == 'RAZE').sum())
            })

    result['lifespan_distribution'] = lifespan_dist_10

    # 4. Lifespan Distribution (5-year bins)
    print("Creating 5-year lifespan distribution...")
    bin_size_5 = 5
    lifespan_dist_5 = []

    for i in range(0, min(max_lifespan + bin_size_5, 300), bin_size_5):
        bin_df = demo_df[(demo_df['lifespan'] >= i) & (demo_df['lifespan'] < i + bin_size_5)]
        if len(bin_df) > 0:
            lifespan_dist_5.append({
                'range': f"{i}-{i + bin_size_5}",
                'EXTDEM': int((bin_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
                'INTDEM': int((bin_df['DEMOLITION_TYPE'] == 'INTDEM').sum()),
                'RAZE': int((bin_df['DEMOLITION_TYPE'] == 'RAZE').sum())
            })

    result['lifespan_distribution_5yr'] = lifespan_dist_5

    # 5. Demolition Types
    print("Processing demolition types...")
    demo_types = []
    for demo_type in ['EXTDEM', 'INTDEM', 'RAZE']:
        count = (demo_df['DEMOLITION_TYPE'] == demo_type).sum()
        demo_types.append({
            'type': demo_type,
            'count': int(count)
        })

    result['demolition_types'] = demo_types

    # 6. NEW: Material-Lifespan-Demolition Analysis for Stacked Bar Charts
    print("Creating material-lifespan-demolition analysis for stacked bar charts...")

    # Clean material types
    demo_df['material_group'] = demo_df['material_type_desc'].fillna('Unknown')
    demo_df['material_group'] = demo_df['material_group'].str.strip()

    # Structure: {demolition_type: {bin_size: {material: {lifespan_range: count}}}}
    material_lifespan_demo = {}

    demolition_types = ['RAZE', 'EXTDEM', 'INTDEM']
    bin_sizes = [10, 20, 25, 30, 50]

    for demo_type in demolition_types:
        print(f"  Processing {demo_type}...")
        material_lifespan_demo[demo_type] = {}
        demo_type_df = demo_df[demo_df['DEMOLITION_TYPE'] == demo_type]

        for bin_size in bin_sizes:
            material_lifespan_demo[demo_type][f'bin_{bin_size}'] = {}

            # Get all materials for this demolition type
            materials = demo_type_df['material_group'].unique()

            for material in materials:
                material_lifespan_demo[demo_type][f'bin_{bin_size}'][material] = {}
                material_df = demo_type_df[demo_type_df['material_group'] == material]

                if len(material_df) > 0:
                    max_life = int(material_df['lifespan'].max())

                    # Create lifespan bins
                    for i in range(0, min(max_life + bin_size, 200), bin_size):
                        if i + bin_size <= 150:
                            bin_label = f'{i}-{i + bin_size} years'
                            count = len(material_df[(material_df['lifespan'] >= i) &
                                                    (material_df['lifespan'] < i + bin_size)])
                        else:
                            bin_label = f'{i}+ years'
                            count = len(material_df[material_df['lifespan'] >= i])

                        if count > 0:
                            material_lifespan_demo[demo_type][f'bin_{bin_size}'][material][bin_label] = int(count)

                        if i >= 150:
                            break

    result['material_lifespan_demo'] = material_lifespan_demo

    # 6b. Location Data for Map
    print("Processing location data for map...")
    # 限制数据量以避免性能问题
    MAX_MAP_POINTS = 5000  # 可调整

    # 为地图准备数据，包含每个建筑的信息
    map_data = []
    map_df = demo_df.head(MAX_MAP_POINTS) if len(demo_df) > MAX_MAP_POINTS else demo_df

    for _, row in map_df.iterrows():
        if pd.notna(row.get('LONGITUDE')) and pd.notna(row.get('LATITUDE')):  
            map_data.append({
                'lat': float(row['LATITUDE']),
                'lon': float(row['LONGITUDE']),
                'demolition_type': row['DEMOLITION_TYPE'],
                'occ_cls': str(row.get('OCC_CLS', 'N/A')),
                'year_built': int(row['year_built']) if pd.notna(row['year_built']) else 'N/A',
                'demolition_date': row['DEMOLITION_DATE'].strftime('%Y-%m-%d') if pd.notna(
                    row['DEMOLITION_DATE']) else 'N/A'
            })

    result['map_data'] = map_data
    print(f"  Prepared {len(map_data)} points for map")

    # 7. Material Type Statistics
    print("Calculating material type statistics...")
    material_stats = []
    for mat in demo_df['material_group'].unique():
        mat_df = demo_df[demo_df['material_group'] == mat]
        if len(mat_df) > 0:
            material_stats.append({
                'material': mat,
                'count': int(len(mat_df)),
                'avg_lifespan': float(mat_df['lifespan'].mean()),
                'median_lifespan': float(mat_df['lifespan'].median()),
                'demolition_breakdown': {
                    'EXTDEM': int((mat_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
                    'INTDEM': int((mat_df['DEMOLITION_TYPE'] == 'INTDEM').sum()),
                    'RAZE': int((mat_df['DEMOLITION_TYPE'] == 'RAZE').sum())
                }
            })

    # Sort by count
    material_stats.sort(key=lambda x: x['count'], reverse=True)
    result['material_stats'] = material_stats

    # 7b. Material Lifespan Raw Data for Boxplot (Original - keeping for backwards compatibility)
    print("Collecting raw lifespan data for boxplot...")
    material_lifespan_raw = {}

    # Get top 15 materials by count for the boxplot
    top_materials = [mat['material'] for mat in material_stats[:15]]

    for material in top_materials:
        mat_df = demo_df[demo_df['material_group'] == material]
        if len(mat_df) > 0:
            # Store as list of floats for JSON serialization
            lifespans = mat_df['lifespan'].dropna().tolist()
            material_lifespan_raw[material] = [float(x) for x in lifespans]

    result['material_lifespan_raw'] = material_lifespan_raw

    # 7c. NEW: Material Lifespan Raw Data by Demolition Type for Boxplot
    print("Collecting raw lifespan data by demolition type for boxplot...")
    material_lifespan_raw_by_demo = {}

    for demo_type in demolition_types:
        print(f"  Processing raw data for {demo_type}...")
        material_lifespan_raw_by_demo[demo_type] = {}
        demo_type_df = demo_df[demo_df['DEMOLITION_TYPE'] == demo_type]

        # Get materials with at least some data for this demolition type
        material_counts = demo_type_df['material_group'].value_counts()
        top_materials_demo = material_counts.head(20).index.tolist()

        for material in top_materials_demo:
            mat_df = demo_type_df[demo_type_df['material_group'] == material]
            if len(mat_df) > 0:
                lifespans = mat_df['lifespan'].dropna().tolist()
                material_lifespan_raw_by_demo[demo_type][material] = [float(x) for x in lifespans]

    result['material_lifespan_raw_by_demo'] = material_lifespan_raw_by_demo

    # 8. Metadata - UPDATED to include 1940 filter info
    result['metadata'] = {
        'year_range': f"{int(demo_df['demolition_year'].min())}-{int(demo_df['demolition_year'].max())}",
        'generated_date': datetime.now().isoformat(),
        'boston_area_cities': boston_cities,
        'total_ma_buildings': int(len(df)),
        'total_boston_demolitions': int(len(demo_df)),
        'data_note': 'Data represents demolitions in the Boston metropolitan area for buildings built in 1940 or later',
        'year_built_filter': '≥1940',
        'year_built_range': f"{int(demo_df['year_built'].min())}-{int(demo_df['year_built'].max())}"
    }

    # 9. City breakdown
    print("Processing city breakdown...")
    city_stats = []
    for city in demo_df['PROP_CITY'].value_counts().head(10).index:
        city_df = demo_df[demo_df['PROP_CITY'] == city]
        city_stats.append({
            'city': city,
            'count': int(len(city_df)),
            'avg_lifespan': float(city_df['lifespan'].mean())
        })

    result['city_stats'] = city_stats

    # Print summary of material-lifespan-demo data
    print("\nMaterial-Lifespan-Demolition Summary:")
    for demo_type in demolition_types:
        total_materials = len(material_lifespan_demo[demo_type]['bin_20'])
        print(f"  {demo_type}: {total_materials} unique materials")

        # Get top 3 materials for this demo type
        material_counts = {}
        for material, lifespans in material_lifespan_demo[demo_type]['bin_20'].items():
            material_counts[material] = sum(lifespans.values())

        top_3 = sorted(material_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for material, count in top_3:
            print(f"    - {material}: {count} buildings")

    # Print summary of boxplot data
    print("\nBoxplot Data Summary:")
    print(f"  Total materials for boxplot: {len(material_lifespan_raw)}")
    for material in list(material_lifespan_raw.keys())[:5]:
        count = len(material_lifespan_raw[material])
        print(f"    - {material}: {count} data points")

    # Print summary of new boxplot data by demolition type
    print("\nBoxplot Data by Demolition Type Summary:")
    for demo_type in demolition_types:
        material_count = len(material_lifespan_raw_by_demo[demo_type])
        print(f"  {demo_type}: {material_count} materials")
        avg_lifespan = material_lifespan_demo_avg[demo_type]
        print(f"    Average lifespan: {avg_lifespan:.1f} years")

    return result


def save_json(data, filename='boston_demolition_data.json'):
    """Save data to JSON file"""
    print(f"\nSaving to {filename}...")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    # Calculate file size
    import os
    file_size = os.path.getsize(filename) / 1024  # KB
    print(f"JSON file created: {filename} ({file_size:.1f} KB)")


def main():
    """Main execution function"""
    print("=" * 50)
    print("Boston Building Demolition Data Processor")
    print("With Material-Lifespan Stacked Bar Analysis")
    print("Boxplot Support with Demolition Type Filtering")
    print("Filtering for buildings built in 1940 or later")
    print("=" * 50)

    try:
        # Process the data
        data = process_demolition_data()

        # Save to JSON
        save_json(data)

        # Print summary
        print("\n" + "=" * 50)
        print("Processing Complete!")
        print("=" * 50)
        print(f"Total Boston demolitions processed: {data['summary_stats']['total_demolitions']:,}")
        print(f"Buildings built between: {data['metadata']['year_built_range']}")
        print(f"Average building lifespan: {data['summary_stats']['average_lifespan']:.1f} years")
        print(f"Date range: {data['metadata']['year_range']}")

        print(f"\nDemolition type breakdown:")
        print(f"  - INTDEM: {data['summary_stats']['intdem_count']:,}")
        print(f"  - EXTDEM: {data['summary_stats']['extdem_count']:,}")
        print(f"  - RAZE: {data['summary_stats']['raze_count']:,}")

        print(f"\nAverage lifespan by demolition type:")
        for demo_type, avg_lifespan in data['material_lifespan_demo_avg'].items():
            print(f"  - {demo_type}: {avg_lifespan:.1f} years")

        print("\nTop 5 cities by demolition count:")
        for city in data['city_stats'][:5]:
            print(f"  - {city['city']}: {city['count']:,} demolitions")

        print("\nTop 5 material types:")
        for mat in data['material_stats'][:5]:
            print(f"  - {mat['material']}: {mat['count']:,} buildings (avg lifespan: {mat['avg_lifespan']:.1f} years)")

        # Boxplot data info
        if 'material_lifespan_raw' in data:
            print(f"\nBoxplot data prepared for {len(data['material_lifespan_raw'])} materials")

        if 'material_lifespan_raw_by_demo' in data:
            print(f"\nBoxplot data by demolition type prepared:")
            for demo_type, materials in data['material_lifespan_raw_by_demo'].items():
                print(f"  - {demo_type}: {len(materials)} materials")

        print("\n" + "=" * 50)
        print("JSON file 'boston_demolition_data.json' has been created.")
        print("Open index.html in your browser to view the dashboard.")
        print("The dashboard will display:")
        print("- Heatmaps and stacked bar charts for each demolition type")
        print("- Boxplots filtered by demolition type")
        print("- Average lifespan statistics by demolition type")
        print("Note: Data filtered for buildings built in 1940 or later")
        print("=" * 50)

    except FileNotFoundError:
        print(f"\nError: Could not find 'ma_structures_with_demolition_FINAL.csv'")
        print("Please ensure the CSV file is in the current directory.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()