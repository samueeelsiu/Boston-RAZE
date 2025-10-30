# Boston Building Demolition Analysis Dashboard

**⚠️ Last Updated: September 24th, 2025 | Currently Not Being Maintained**

An interactive web-based visualization dashboard for analyzing building demolition patterns in the Greater Boston area, focusing on buildings constructed from 1940 onwards. This tool provides comprehensive analysis of demolition trends, material lifespans, and urban development patterns using data from the NSI Enhanced USA Structures Dataset.

## Live Demo

[View Live Dashboard](https://samueeelsiu.github.io/Boston-RAZE/)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Data Pipeline](#data-pipeline)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Data Structure](#data-structure)
- [Methodology](#methodology)
- [Technologies](#technologies)
- [Support](#support)

## Overview

This dashboard visualizes and analyzes building demolition data from the Greater Boston metropolitan area, integrating:
- **NSI Enhanced USA Structures (MA)**: Massachusetts building inventory
- **Property Assessment Data**: Building characteristics and materials
- **Demolition Permits**: Historical demolition records
- **Focus Period**: Buildings constructed 1940 or later

The analysis covers demolitions across Boston, Cambridge, Somerville, Brookline, Quincy, Newton, Watertown, Chelsea, Revere, and Everett.

## Key Features

### Interactive Analysis Components

#### 1. **Demolition Type Filtering**
- Three demolition categories:
  - **RAZE**: Complete structure demolition
  - **EXTDEM**: Exterior demolition
  - **INTDEM**: Interior demolition
- Radio button controls for instant filtering
- Type-specific statistics and visualizations

#### 2. **Material-Lifespan Heatmap**
- Interactive heatmap showing demolition counts by material and lifespan
- Plasma colormap with linear scale
- Top 15 materials by demolition count
- Configurable lifespan bin sizes (10, 20, 25, 30, 50 years)
- Toggle between count and percentage display modes

#### 3. **Stacked Bar Charts**
- Material type vs lifespan distribution
- Separate charts for each demolition type
- Dynamic color coding (darkest for largest values)
- Percentage and count view options

#### 4. **Geospatial Mapping**
- Interactive Leaflet map with OpenStreetMap tiles
- Color-coded markers by demolition type
- Hover tooltips with building details
- Up to 5,000 mapped demolition points

#### 5. **Temporal Analysis**
- Yearly demolition trends
- Stacked or individual demolition type views
- Historical patterns visualization

#### 6. **Building Lifespan Distribution**
- Histogram of building lifespans
- 10-year bin analysis
- Demolition type breakdown

#### 7. **Material Statistics Table**
- Top materials by demolition count
- Average lifespan per material
- Demolition type breakdown per material

#### 8. **Boxplot Analysis**
- Building lifespan distribution by material type
- Quartile visualization
- Outlier detection
- Filterable by demolition type

#### 9. **City-Level Statistics**
- Demolition counts by city
- Grid layout for easy comparison
- Top 10 cities by demolition activity

## Data Pipeline

### Processing Workflow

```
Stage 1: Data Filtering
├── Input: NSI Enhanced USA Structures (MA) dataset
├── Filter 1: Extract demolition records only
├── Filter 2: Boston metro area cities
├── Filter 3: Buildings built ≥1940
└── Output: Filtered demolition dataset

Stage 2: Data Cleaning
├── Date conversion and validation
├── Lifespan calculation (demolition year - built year)
├── Remove invalid lifespans (<0 or >500 years)
├── Deduplicate by BUILD_ID
└── Output: Clean unique building records

Stage 3: Analysis Generation
├── Summary statistics calculation
├── Material-lifespan matrix creation
├── Temporal aggregation
├── Geospatial data extraction
└── Output: JSON data structure

Stage 4: Visualization Preparation
├── Bin size variations (10, 20, 25, 30, 50 years)
├── Percentage calculations
├── Color mapping
└── Output: boston_demolition_data.json
```

## Installation

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/samueeelsiu/Boston-RAZE.git
cd Boston-RAZE
```

2. **File Structure Required**
```
boston-demolition-analysis/
├── index.html                          # Main dashboard
├── boston_demolition_data.json         # Processed demolition data
├── preprocessor.py                     # Data processing script
└── README.md                          # This file
```

3. **Launch the Dashboard**

Option A: GitHub Pages
```bash
# Navigate to https://[your-username].github.io/[repository-name]/
```

Option B: Local server
```bash
# Python 3
python -m http.server 8000

# Node.js
npx http-server

# Then navigate to http://localhost:8000
```

### Data Processing (Optional)

To regenerate the data file from source:

```bash
# Install dependencies
pip install pandas numpy

# Ensure source file is present
# ma_structures_with_demolition_FINAL.csv

# Run the preprocessor
python preprocessor.py
```

## Usage Guide

### Controls

1. **Demolition Type Selection**: Use radio buttons to filter by demolition type
2. **Lifespan Bin Size**: Adjust granularity of lifespan analysis (10-50 years)
3. **Display Mode**: Toggle between count and percentage views
4. **Visualization Toggle**: Switch between heatmap and stacked bar chart views

### Key Interactions

- **Heatmap**: Hover for values, click for details
- **Map**: Pan and zoom, hover markers for building information
- **Charts**: Hover for tooltips, click legends to show/hide series
- **Boxplot**: View quartiles and outliers for material lifespans

## Data Structure

### JSON Schema Overview

```javascript
{
  "summary_stats": {
    "total_demolitions": ...,
    "average_lifespan(RAZE)": 58.6 yrs,
    "extdem_count": 285,
    "intdem_count": 800,
    "raze_count": 151,
    "min_year_built": 1939,
    "max_year_built": 2024
  },
  "material_lifespan_demo_avg": {
    "RAZE": ...,
    "EXTDEM": ...,
    "INTDEM": ...
  },
  "yearly_stacked": [...],              // Annual demolition counts
  "lifespan_distribution": [...],       // 10-year bins
  "material_lifespan_demo": {...},      // Core heatmap/chart data
  "material_stats": [...],              // Material statistics
  "material_lifespan_raw": {...},       // Raw data for boxplot
  "material_lifespan_raw_by_demo": {...}, // Boxplot by demo type
  "map_data": [...],                    // Geospatial points
  "city_stats": [...],                  // City-level statistics
  "metadata": {
    "year_range": ...,
    "generated_date": ...,
    "boston_area_cities": ...,
    "total_ma_buildings": ...,
    "total_boston_demolitions": ...,
    "year_built_filter": "≥1940",
    "year_built_range": string
  }
}
```

## Methodology

### Key Analytical Approaches

1. **Lifespan Calculation**: Demolition year minus construction year
2. **Material Grouping**: Standardized material type classification
3. **Binning Strategy**: Multiple bin sizes for flexible analysis
4. **Color Mapping**: Plasma colormap for intuitive data visualization
5. **Statistical Methods**: 
   - Mean/median calculations
   - Quartile analysis for boxplots
   - Percentage normalization

### Data Quality Measures

- Removal of pre-1940 buildings for consistency
- Duplicate removal based on BUILD_ID
- Invalid lifespan filtering
- Null value handling

## Technologies

### Frontend
- **HTML5/CSS3**: Responsive layout with modern styling
- **JavaScript ES6+**: Dynamic interactions and data processing
- **Chart.js v4.4.0**: Primary charting library
- **Leaflet v1.9.4**: Interactive mapping
- **chartjs-chart-boxplot v4.2.7**: Boxplot visualizations

### Backend Processing
- **Python 3.x**
  - pandas: Data manipulation and analysis
  - numpy: Numerical computations
  - datetime: Temporal data handling
  - json: Data serialization

### Data Formats
- **JSON**: Primary data exchange format
- **CSV**: Source data format

## Performance Considerations

### Optimizations

1. **Map Point Limiting**: Maximum 5,000 points for performance
2. **Pre-computed Aggregations**: All statistics pre-calculated
3. **Efficient Data Structure**: Hierarchical JSON for quick access
4. **Dynamic Chart Updates**: Only re-render changed visualizations

### Known Limitations

- Map visualization limited to 5,000 points
- Building year filter excludes pre-1940 structures
- Demolition data specific to Greater Boston area only

## Credit

### Development Team
- **Developer**: Lang (Samuel) Shao
- **Supervisor**: Prof. Demi Fang
- **Institution**: [Northeastern University](https://www.northeastern.edu/)
- **Lab**: [Structural Futures Lab](https://structural-futures.org/)

### Data Sources
- NSI Enhanced USA Structures Dataset
- Massachusetts Property Assessment Database
- Approved Building Permits From City of Boston

## Support

For issues, questions, or suggestions regarding this dashboard, please contact: shao.la@northeastern.edu

**Note**: This project is no longer actively maintained as of September 24th, 2025. The dashboard remains functional but will not receive further updates or enhancements.
