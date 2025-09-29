# Airline Customer Value Analysis System

## Project Overview

This project utilizes the K-Means clustering algorithm to analyze airline customer data, helping airlines identify different customer segments through customer value analysis. This enables the development of targeted marketing strategies and service solutions. The system analyzes multidimensional data including customer flight behavior, consumption patterns, and membership characteristics to categorize customers into different value groups, providing data support for the airline's customer relationship management.

## Key Features

- **Data Preprocessing**: Automatically cleans and processes raw customer data, including removing invalid data and handling missing values
- **Feature Engineering**: Calculates key customer features such as membership duration, average price per kilometer, flight behavior stability, etc.
- **Cluster Analysis**: Uses K-Means algorithm to segment customers and determines the optimal number of clusters through SSE (Sum of Squared Errors)
- **Visualization**: Intuitively displays feature differences between customer segments through radar charts
- **Flexible Configuration**: Supports adjustment of clustering parameters to try different clustering schemes (K=4,5,6)

## Data Description

The dataset `air_data.csv` used in this project contains detailed information about airline customers, with key fields including:

- **MEMBER_NO**: Membership number
- **FFP_DATE**: Membership registration date
- **FLIGHT_COUNT**: Number of flights
- **SUM_YR_1**: Total ticket price in the first year
- **SUM_YR_2**: Total ticket price in the second year
- **SEG_KM_SUM**: Total flight kilometers within the observation window
- **AVG_INTERVAL**: Average time interval between flights
- **MAX_INTERVAL**: Maximum flight interval within the observation window
- **avg_discount**: Average discount rate
- Other basic member information (gender, age, work city, etc.)

## Core Features

The system selects the following six core features for customer clustering:

1. **Membership Duration**: Reflects the length of customer membership history
2. **Flight Count**: Reflects customer activity level
3. **Average Price per Kilometer**: Reflects customer spending level
4. **Total Mileage**: Reflects the sum of customer flight distances
5. **Time Interval Difference**: Reflects the regularity of customer flight behavior (difference between maximum interval and average interval)
6. **Average Discount Rate**: Reflects customer price sensitivity

## Usage Instructions

### Requirements

- Python 3.6+
- Dependencies: pandas, numpy, matplotlib, scikit-learn

### Installing Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Running the Program

```bash
python custom_value.py
```

### Output Results

After running the program, the following results will be generated:

1. Information output from the data preprocessing process
2. SSE value curve for different K values (number of clusters), used to determine the optimal number of clusters
3. Customer segment feature radar charts for three clustering schemes: K=4, K=5, and K=6

## Algorithm Description

### K-Means Clustering

K-Means is a commonly used clustering algorithm that divides data points into K clusters through iterative optimization. In this project:

1. Data is first standardized (Z-score standardization)
2. The optimal number of clusters is determined by calculating the SSE (Sum of Squared Errors) for different K values
3. Radar charts are used to visualize feature differences between different customer segments

### SSE (Sum of Squared Errors)

SSE is an important evaluation metric in the K-Means clustering algorithm, calculating the sum of squared distances from each data point to its cluster center. A smaller SSE indicates better clustering results. By plotting the SSE curve as K values change, the optimal number of clusters can be found (usually at the "elbow" of the curve).

## Project Structure

```
KMeans_Air_Customer/
├── custom_value.py      # Main program code
├── data/                # Data directory
│   ├── air_data.csv     # Airline customer data
│   └── Kmeans聚类客户价值分析.pdf  # Project documentation
├── README_EN.md         # English documentation
└── README_CN.md         # Chinese documentation
```

## Interpreting Results

The radar charts provide an intuitive view of the differences between customer segments across the six core features:

- Each radar chart represents a customer segment
- Each axis of the radar chart represents a feature dimension
- The area size of the chart reflects the overall performance of the segment across various features
- Based on the feature differences between segments, targeted marketing strategies can be developed

## Future Optimization Directions

1. Incorporate more customer behavior features to improve clustering accuracy
2. Try other clustering algorithms (such as DBSCAN, hierarchical clustering, etc.) for comparative analysis
3. Develop an interactive visualization interface to enhance user experience
4. Add customer value prediction functionality to forecast future customer value
5. Integrate the RFM model (Recency, Frequency, Monetary) for more comprehensive customer value assessment

## Notes

- This project sets the number of CPU cores to 4 on Windows 11 systems to resolve the potential "UserWarning: Could not find the number of physical cores" error
- Unreasonable data records, such as records with zero ticket price but non-zero flight kilometers, are removed during data preprocessing
- The radar charts use Chinese fonts, so please ensure that the SimHei font is installed on your system