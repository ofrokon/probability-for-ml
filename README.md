# Probability Theory for Machine Learning

This repository contains Python scripts demonstrating key concepts in probability theory that are essential for machine learning. It accompanies the Medium post "Introduction to Probability Theory for Machine Learning".

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Visualizations](#visualizations)
4. [Topics Covered](#topics-covered)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

To run these scripts, you need Python 3.6 or later. Follow these steps to set up your environment:

1. Clone this repository:
   ```
   git clone https://github.com/ofrokon/probability-for-ml.git
   cd probability-for-ml
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To generate all visualizations, run:

```
python create_visualizations.py
```

This will create PNG files for each visualization in the current directory.

## Visualizations

This script generates the following visualizations:

1. `conditional_probability.png`: Illustrates conditional probability using coin flip examples.
2. `binomial_distribution.png`: Shows the probability mass function of a binomial distribution.
3. `normal_distribution.png`: Displays the probability density function of a normal distribution.
4. `maximum_likelihood_estimation.png`: Demonstrates maximum likelihood estimation for a normal distribution.

## Topics Covered

- Basic Probability Concepts
- Conditional Probability and Independence
- Probability Distributions (Discrete and Continuous)
- Bayes' Theorem
- Maximum Likelihood Estimation

Each topic is explained in the accompanying Medium post with Python code examples.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your suggested changes.

## License

This project is open source and available under the [MIT License](LICENSE).

---

For a detailed explanation of these probability concepts and their application in machine learning, check out the accompanying Medium post: [Introduction to Probability Theory for Machine Learning](https://medium.com/@mroko001/introduction-to-probability-theory-for-machine-learning-97584f8e585b)

For questions or feedback, please open an issue in this repository.
