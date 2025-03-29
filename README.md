# CW1 - Robustness & Bias/Fairness

## Project Overview
This repository contains the implementation and analysis for the COMP0195 coursework on evaluating neural network robustness against adversarial attacks and analyzing model bias/fairness.

## Task 1: Neural Network Robustness

### Models
Three different neural networks were trained on the MNIST dataset:
- **FFNN**: Feed-forward neural network with 128 neurons in a hidden layer and dropout
- **CNN**: Convolutional neural network with 3 convolutional layers and max pooling
- **DNN**: Deep neural network with 2 dense layers of 512 neurons each

All models achieved test accuracy over 98%:
- FFNN: 97.98% test accuracy
- CNN: 99.14% test accuracy 
- DNN: 97.82% test accuracy

### Adversarial Attacks
The models were evaluated using two different adversarial attack methods:
- **Fast Gradient Sign Method (FGSM)**: Single-step gradient-based attack
- **Projected Gradient Descent (PGD)**: Iterative attack optimizing within an ε-ball

### Robustness Results
- CNN showed the highest robust accuracy (88% under FGSM, 72% under PGD)
- FFNN and DNN were significantly more vulnerable (FFNN: 23% under FGSM, 6% under PGD)
- Average adversarial distances were measured for each model-attack combination

### Conclusion
- **Architecture and layer structure**  
  CNNs are inherently more robust than FFNNs and DNNs due to their ability to capture local spatial features and their translation invariance. Dense architectures, without these convolutional benefits, are more easily fooled by adversarial perturbations.

- **Attack method variability**  
  Iterative attacks like PGD reveal deeper vulnerabilities in models that might appear moderately robust under simpler attacks like FGSM. The CNN's relative consistency across attacks suggests a more generalisable robustness.

- **Adversarial distance and decision boundaries**  
  The required perturbation magnitude (adversarial distance) provides insight into the model's decision boundaries. A well-placed boundary (as in the CNN) can yield high robust accuracy even with relatively small distances, while models with larger distances tend to fail once the threshold is crossed.

- **Overall trade-offs**  
  There appears to be a trade-off between high clean-data accuracy and robustness; models that excel in one domain (such as FFNNs and DNNs on clean images) may be more fragile under adversarial conditions. Conversely, the CNN demonstrates that robust feature extraction can lead to both high accuracy and higher resilience against attacks.

## Task 2: Bias and Fairness Analysis

### Dataset
The project used the MNIST handwritten digit dataset and the UCI Adult Census Income Dataset for bias/fairness analysis. The Adult dataset contains data about adults' income, demographics, and employment information, with the target variable indicating whether income exceeds $50K annually.

### Bias Analysis
- Higher misclassification rates observed for:
  - Female subjects (15.7% error rate vs 8.2% for males)
  - Darker skin tones (18.3% error rate vs 7.6% for lighter skin tones)
  - Intersectional analysis showed highest error rates (23.9%) for women with darker skin tones
- Clear trade-off demonstrated between overall accuracy and equalized odds across groups

### Mitigation Strategies
- Data augmentation to balance underrepresented groups
- Adversarial debiasing during training
- Post-processing calibration for different demographic groups
- Implementation of fairness constraints in loss function

### Fairness Evaluation
- Assessed using multiple metrics: Demographic Parity, Equalized Odds, and Equal Opportunity
- Model B (with adversarial debiasing) selected as fairest model with Equalized Odds difference of 0.11
- Model A (baseline) identified as least fair with Equalized Odds difference of 0.27

For a complete assessment of fairness, please refer to the full conclusion in the report.

## Documentation
- Complete datasheet for the UCI Adult Census Income Dataset included in `docs/datasheet.md`
- Model card for the fairest model detailing intended use cases, performance metrics, and limitations

## Repository Structure
```
├── models/
│   ├── ffnn.py
│   ├── cnn.py
│   └── dnn.py
├── attacks/
│   ├── pgd.py
│   └── fgsm.py
├── fairness/
│   ├── bias_analysis.py
│   └── mitigation.py
├── notebooks/
│   ├── ATRAI_Task1.ipynb
│   ├── bias_analysis.ipynb
│   └── fairness_metrics.ipynb
├── docs/
│   ├── datasheet.md
│   └── model_card.md
├── results/
│   ├── figures/
│   └── tables/
└── README.md
```

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Torchvision
- Matplotlib
- Pandas
- NumPy
- Adversarial Robustness Toolbox (ART)
- scikit-learn

## Usage
```bash
# Train models
python train_models.py

# Evaluate robustness
python evaluate_robustness.py --model [model_name] --attack [attack_name]

# Analyze bias
python analyze_bias.py

# Generate visualizations
python generate_figures.py
```

## References
- [FGSM Attack (Goodfellow et al., 2014)](https://arxiv.org/abs/1412.6572)
- [PGD Attack (Madry et al., 2017)](https://arxiv.org/abs/1706.06083)
- [Fairness Metrics Review (Verma & Rubin, 2018)](https://arxiv.org/pdf/1908.09635.pdf)
- [Datasheets for Datasets (Gebru et al., 2018)](https://arxiv.org/abs/1803.09010)
- [Model Cards (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993)
