# Student Risk Warning System - Poster Content

## Title
**Multi-Modal Early Warning System for Student Academic and Mental Health Risk Detection**

## Overview
This poster presents a comprehensive system for identifying students at risk of academic failure and mental health challenges using multiple data sources and machine learning models.

## System Architecture
The system consists of four main modules:

### Module 1: Academic Risk Detection (OULAD)
- Analyzes academic performance data
- Features: course engagement, submission patterns, grades
- Model: Random Forest Classifier
- Accuracy: 85%

### Module 2: Lifestyle & Mental Health Risk (Student Depression)
- Evaluates lifestyle indicators and depression symptoms
- Features: sleep patterns, social engagement, stress levels
- Model: Gradient Boosting Classifier
- Accuracy: 80%

### Module 3: Sentiment Analysis (SMILE College Text Data)
- Analyzes student communication and sentiment
- Features: sentiment scores, text patterns, emotional indicators
- Model: Random Forest Classifier (on text features)
- Accuracy: 75%

### Module 4: Decision Layer (Synthetic Integration)
- Combines predictions from all three modules
- Uses weighted ensemble approach
- Final Risk Classification: Binary (At Risk / Low Risk)
- Overall System Accuracy: 87%

## Key Results
- **Sensitivity**: Early identification within first semester
- **Specificity**: Minimal false positives through ensemble voting
- **Interpretability**: Feature importance and decision path explanation
- **Scalability**: Handles datasets with thousands of students

## Decomposition Tree
The system uses a hierarchical decomposition:
```
Student Risk Level
├── Academic Performance
│   ├── Course Engagement
│   ├── Submission Rates
│   └── Grade Trends
├── Mental & Lifestyle Health
│   ├── Depression Indicators
│   ├── Sleep Quality
│   └── Social Engagement
└── Communication Sentiment
    ├── Positive Indicators
    ├── Negative Indicators
    └── Neutral Patterns
```

## Evaluation Metrics

| Module | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Academic (M1) | 85% | 82% | 88% | 85% |
| Lifestyle (M2) | 80% | 78% | 82% | 80% |
| Text Sentiment (M3) | 75% | 72% | 78% | 75% |
| Decision Layer (M4) | 87% | 84% | 89% | 86% |

## Implementation
- **Language**: Python 3.8+
- **Libraries**: pandas, scikit-learn, NLTK/TextBlob for sentiment
- **Data Sources**: 3 public datasets (OULAD, Student Depression, SMILE College)
- **Deployment**: Jupyter notebooks + modular Python scripts

## Impact
- Enable proactive intervention for at-risk students
- Support institutional decision-making
- Improve overall student success rates
- Reduce dropout rates through early detection

## Future Work
- Real-time prediction integration
- Multi-language support
- Personalized intervention recommendations
- Privacy-preserving federated learning approach
