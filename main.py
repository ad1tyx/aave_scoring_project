import sys
import os

# This adds the 'src' folder to Python's path to find our code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from scorer import CreditScorer

if __name__ == '__main__':
    print("🚀 Starting the Aave Credit Scoring Pipeline...")
    
    INPUT_FILE = os.path.join('data', 'user-wallet-transactions.json')
    
    # Create an instance of the scorer and run the pipeline
    credit_scorer = CreditScorer(input_filepath=INPUT_FILE)
    credit_scorer.run()