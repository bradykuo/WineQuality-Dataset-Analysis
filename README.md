# Wine Quality Dataset Analysis 

This project analyzes wine quality data from the UCI Machine Learning Repository, focusing on both red and white Vinho Verde wines from Portugal. The analysis includes both binary classification (red vs. white wine) and multi-class classification (quality levels) using various machine learning models.<br>
<br>
（成大統計系｜機器學習｜作業）

## Dataset Description
The dataset contains physical-chemical properties of wines:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (score between 0 and 10)
- Type (red or white)

## Analysis Components

### 1. Binary Classification
Classifying wines as either red or white using:
- Logistic Regression
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- K-Nearest Neighbors (KNN)
- Random Forest

### 2. Multi-class Classification
Predicting wine quality levels (Low, Medium, High) for both red and white wines separately using the same models.

## Implementation

### Prerequisites
```R
# R packages
library(tidyverse)
library(caret)
library(MASS)
library(class)
library(randomForest)
library(nnet)

# Python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
```

### File Structure
```
├── data/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── winequality.csv
├── src/
│   ├── Binary_Classification.R
│   ├── Multi-class_Classification.R
│   └── winequality_clustering_analysis.py
└── README.md
```

## Key Findings

### Binary Classification Results
- All models achieved over 90% accuracy in distinguishing between red and white wines
- KNN showed slightly lower performance compared to other models
- Random Forest and LDA demonstrated the highest accuracy

### Multi-class Classification Results
Red Wine:
- Random Forest: 72.54%
- LDA: 62.86%
- Logistic Regression: 61.79%
- QDA: 60.23%
- KNN: 52.35%

White Wine:
- Random Forest: 72.68%
- LDA: 57.51%
- Logistic Regression: 56.78%
- QDA: 52.92%
- KNN: 51.08%

## Usage

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required packages:
```R
# In R
install.packages(c("tidyverse", "caret", "MASS", "class", "randomForest", "nnet"))
```
```python
# In Python
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Run the analysis:
```R
# For binary classification
source("src/Binary_Classification.R")

# For multi-class classification
source("src/Multi-class_Classification.R")
```
```python
# For clustering analysis
python src/winequality_clustering_analysis.py
```

## Data Source
The wine quality dataset is from the UCI Machine Learning Repository:
http://archive.ics.uci.edu/ml/datasets/Wine+Quality

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is available for academic and educational purposes.
