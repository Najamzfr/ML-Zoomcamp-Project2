# Step 6: Portfolio Optimization
def optimize_portfolio(predictions):
    print("Optimizing portfolio...")

    # Combine all predicted returns into a DataFrame
    pred_returns = pd.DataFrame(predictions)
    mean_returns = pred_returns.mean()  # Expected returns
    cov_matrix = pred_returns.cov()  # Covariance matrix

    # Objective function: minimize negative Sharpe ratio
    def neg_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        return -sharpe_ratio

    # Constraints: weights sum to 1
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(len(mean_returns)))

    # Initial guess
    initial_weights = [1 / len(mean_returns)] * len(mean_returns)

    # Optimization
    result = minimize(neg_sharpe, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    print("Optimal portfolio weights:", optimal_weights)

    # Plot efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_returns.std(), mean_returns, label="Individual Stocks")
    plt.scatter(np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))),
                np.dot(optimal_weights, mean_returns), c="red", label="Optimal Portfolio", marker="X", s=100)
    plt.title("Efficient Frontier")
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Return")
    plt.legend()
    plt.grid()
    plt.savefig("reports/figures/efficient_frontier.png")
    plt.close()