import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def process_demolition_data(csv_path='ma_structures_with_demolition_FINAL.csv'):
    """
    Process MA structures data focusing on Boston demolitions and output JSON for dashboard
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
        'raze_count': int((demo_df['DEMOLITION_TYPE'] == 'RAZE').sum())
    }

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

    # 6. Material Type Analysis for Sankey Diagram (NO AGGREGATION)
    print("Processing material type analysis (NO AGGREGATION - keeping original types)...")

    # Use original material types without any aggregation
    demo_df['material_group'] = demo_df['material_type_desc'].fillna('Unknown')

    # Just basic cleaning - remove extra spaces and standardize case
    demo_df['material_group'] = demo_df['material_group'].str.strip()

    # Optional: You might want to see what material types you actually have
    print(f"Found {demo_df['material_group'].nunique()} unique material types")
    print("Top 10 material types:")
    print(demo_df['material_group'].value_counts().head(10))

    # Create detailed sankey data for dynamic filtering
    sankey_detailed = {
        'materials': list(demo_df['material_group'].unique()),
        'demolition_types': ['EXTDEM', 'INTDEM', 'RAZE'],
        'material_to_demo': {},
        'demo_to_lifespan': {}
    }

    # Calculate material to demolition flows
    for mat in sankey_detailed['materials']:
        sankey_detailed['material_to_demo'][mat] = {}
        for demo_type in sankey_detailed['demolition_types']:
            count = len(demo_df[(demo_df['material_group'] == mat) &
                                (demo_df['DEMOLITION_TYPE'] == demo_type)])
            sankey_detailed['material_to_demo'][mat][demo_type] = int(count)

    # Calculate demolition to lifespan flows for different bin sizes
    bin_sizes = [10, 20, 25, 30, 50]
    for bin_size in bin_sizes:
        sankey_detailed['demo_to_lifespan'][f'bin_{bin_size}'] = {}

        for demo_type in sankey_detailed['demolition_types']:
            sankey_detailed['demo_to_lifespan'][f'bin_{bin_size}'][demo_type] = {}
            demo_type_df = demo_df[demo_df['DEMOLITION_TYPE'] == demo_type]

            # Create bins
            for i in range(0, 200, bin_size):
                if i + bin_size <= 150:
                    bin_label = f'{i}-{i + bin_size} years'
                else:
                    bin_label = f'{i}+ years'

                count = len(demo_type_df[(demo_type_df['lifespan'] >= i) &
                                         (demo_type_df['lifespan'] < i + bin_size)])
                if count > 0:
                    sankey_detailed['demo_to_lifespan'][f'bin_{bin_size}'][demo_type][bin_label] = int(count)

                if i >= 150:
                    break

    result['sankey_detailed'] = sankey_detailed

    # Also keep the original simple sankey for backwards compatibility
    sankey_data = {
        'nodes': [],
        'links': []
    }

    # (Keep original sankey_data code here for compatibility...)
    materials = demo_df['material_group'].unique()
    demo_types_list = ['EXTDEM', 'INTDEM', 'RAZE']

    node_idx = 0
    node_map = {}

    for mat in materials:
        sankey_data['nodes'].append({'name': mat, 'category': 'material'})
        node_map[f'mat_{mat}'] = node_idx
        node_idx += 1

    for demo_type in demo_types_list:
        sankey_data['nodes'].append({'name': demo_type, 'category': 'demolition'})
        node_map[f'demo_{demo_type}'] = node_idx
        node_idx += 1

    # Default 25-year bins for simple sankey
    for i in range(0, 175, 25):
        if i < 150:
            label = f'{i}-{i + 25} years'
        else:
            label = '150+ years'
        sankey_data['nodes'].append({'name': label, 'category': 'lifespan'})
        node_map[f'life_{label}'] = node_idx
        node_idx += 1

    # Add links
    for mat in materials:
        for demo_type in demo_types_list:
            count = sankey_detailed['material_to_demo'][mat][demo_type]
            if count > 0:
                sankey_data['links'].append({
                    'source': node_map[f'mat_{mat}'],
                    'target': node_map[f'demo_{demo_type}'],
                    'value': count
                })

    for demo_type in demo_types_list:
        demo_type_df = demo_df[demo_df['DEMOLITION_TYPE'] == demo_type]
        for i in range(0, 175, 25):
            if i < 150:
                label = f'{i}-{i + 25} years'
                count = len(demo_type_df[(demo_type_df['lifespan'] >= i) &
                                         (demo_type_df['lifespan'] < i + 25)])
            else:
                label = '150+ years'
                count = len(demo_type_df[demo_type_df['lifespan'] >= 150])

            if count > 0:
                sankey_data['links'].append({
                    'source': node_map[f'demo_{demo_type}'],
                    'target': node_map[f'life_{label}'],
                    'value': int(count)
                })

    result['sankey_data'] = sankey_data

    # 7. Material Type Statistics
    print("Calculating material type statistics...")
    material_stats = []
    for mat in demo_df['material_group'].unique():
        mat_df = demo_df[demo_df['material_group'] == mat]
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

    # 8. Metadata
    result['metadata'] = {
        'year_range': f"{int(demo_df['demolition_year'].min())}-{int(demo_df['demolition_year'].max())}",
        'generated_date': datetime.now().isoformat(),
        'boston_area_cities': boston_cities,
        'total_ma_buildings': int(len(df)),
        'total_boston_demolitions': int(len(demo_df)),
        'data_note': 'Data represents demolitions in the Boston metropolitan area only, extracted from Massachusetts statewide building dataset'
    }

    # 9. City breakdown (for Boston area)
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

    return result


def save_json(data, filename='boston_demolition_data.json'):
    """Save data to JSON file"""
    print(f"Saving to {filename}...")
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
        print(f"Average building lifespan: {data['summary_stats']['average_lifespan']:.1f} years")
        print(f"Date range: {data['metadata']['year_range']}")
        print(f"\nDemolition type breakdown:")
        print(f"  - INTDEM: {data['summary_stats']['intdem_count']:,}")
        print(f"  - EXTDEM: {data['summary_stats']['extdem_count']:,}")
        print(f"  - RAZE: {data['summary_stats']['raze_count']:,}")

        print("\nTop 5 cities by demolition count:")
        for city in data['city_stats'][:5]:
            print(f"  - {city['city']}: {city['count']:,} demolitions")

        print("\nJSON file 'boston_demolition_data.json' has been created.")
        print("Upload both this JSON file and your HTML dashboard to GitHub.")

    except Exception as e:
        print(f"\nError: {e}")
        print("Please ensure 'ma_structures_with_demolition_FINAL.csv' is in the current directory.")
        raise


if __name__ == "__main__":
    main()