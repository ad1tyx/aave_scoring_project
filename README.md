# Aave V2 Credit Scoring Model

This project implements a machine learning model to assign a credit score (0-1000) to wallet addresses based on their historical transaction behavior on the Aave V2 protocol.

## Methodology

The credit score is generated using an **unsupervised, feature-based heuristic model**. This approach relies on engineering features that logically correlate with responsible and risky financial behavior. These features are then combined in a weighted formula to produce a final score.

### Processing Flow

The project is run via a main entrypoint `main.py` which uses a `CreditScorer` class from the `src` directory. The flow is as follows:

1.  **Data Loading & Parsing:** The raw JSON transaction data is loaded and parsed into a clean DataFrame.
2.  **Feature Engineering:** The script aggregates data for each `userAddress` to create a behavioral profile. Key features include:
    * **Account Age:** Duration between the first and last transaction.
    * **Liquidation Count:** The number of times a user was liquidated (heavily penalized).
    * **Repayment Ratio:** The ratio of total USD value repaid versus borrowed.
    * **LTV Proxy:** The ratio of total USD value borrowed versus deposited.
3.  **Score Calculation:** A weighted formula calculates a raw score.
4.  **Normalization:** Scores are scaled to a **0 to 1000** range.
5.  **Output:** Scores are saved to `wallet_scores.csv`, and a distribution graph is saved as `score_distribution.png`.

### How to Run

1.  **Prerequisites:** Python 3.8+.
2.  **Setup:**
    ```bash
    # Create and activate virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```
3.  **Run the script:**
    ```bash
    python main.py
    ```
