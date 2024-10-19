import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, norm
from scipy.optimize import minimize_scalar

def save_plot(fig, filename):
    plt.show()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 1. Conditional Probability
def plot_conditional_probability():
    np.random.seed(42)
    flips = np.random.choice(['H', 'T'], size=(10000, 2))
    prob_both_heads = np.mean((flips[:, 0] == 'H') & (flips[:, 1] == 'H'))
    prob_second_head = np.mean(flips[:, 1] == 'H')
    prob_head_given_head = prob_both_heads / np.mean(flips[:, 0] == 'H')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['P(HH)', 'P(H)', 'P(H|H)'], [prob_both_heads, prob_second_head, prob_head_given_head])
    ax.set_ylim(0, 1)
    ax.set_title("Conditional Probability of Coin Flips")
    ax.set_ylabel("Probability")
    
    for i, v in enumerate([prob_both_heads, prob_second_head, prob_head_given_head]):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center')

    save_plot(fig, 'conditional_probability.png')

# 2. Binomial Distribution
def plot_binomial_distribution():
    n, p = 10, 0.5
    binomial_rv = binom(n, p)
    x = np.arange(0, n+1)
    pmf = binomial_rv.pmf(x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, pmf)
    ax.set_title(f"Binomial Distribution (n={n}, p={p})")
    ax.set_xlabel("Number of successes")
    ax.set_ylabel("Probability")
    save_plot(fig, 'binomial_distribution.png')

# 3. Normal Distribution
def plot_normal_distribution():
    mu, sigma = 0, 1
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    pdf = norm.pdf(x, mu, sigma)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, pdf)
    ax.set_title(f"Normal Distribution (μ={mu}, σ={sigma})")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability Density")
    save_plot(fig, 'normal_distribution.png')

# 4. Maximum Likelihood Estimation
def plot_mle():
    np.random.seed(42)
    true_mu, true_sigma = 5, 2
    data = np.random.normal(true_mu, true_sigma, 1000)

    def neg_log_likelihood(mu, data, sigma):
        return -np.sum(norm.logpdf(data, mu, sigma))

    result = minimize_scalar(neg_log_likelihood, args=(data, true_sigma))
    mle_mu = result.x

    x = np.linspace(true_mu - 3*true_sigma, true_mu + 3*true_sigma, 100)
    true_pdf = norm.pdf(x, true_mu, true_sigma)
    mle_pdf = norm.pdf(x, mle_mu, true_sigma)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=30, density=True, alpha=0.7, label='Data')
    ax.plot(x, true_pdf, 'r-', label=f'True (μ={true_mu})')
    ax.plot(x, mle_pdf, 'g--', label=f'MLE (μ={mle_mu:.2f})')
    ax.set_title("Maximum Likelihood Estimation")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    save_plot(fig, 'maximum_likelihood_estimation.png')

if __name__ == "__main__":
    plot_conditional_probability()
    plot_binomial_distribution()
    plot_normal_distribution()
    plot_mle()
    print("All visualizations have been saved as PNG files.")