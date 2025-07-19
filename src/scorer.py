import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
from decimal import Decimal
import os

class CreditScorer:
    """
    A class to load Aave transaction data, engineer features,
    and calculate a credit score for each wallet.
    """
    def __init__(self, input_filepath, output_dir='.'):
        self.input_filepath = input_filepath
        self.output_dir = output_dir
        self.df = None
        self.wallet_scores = None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"✅ Scorer initialized. Output will be saved to '{self.output_dir}'")

    def load_and_parse_data(self):
        print(f"\n--- Step 1: Loading & Parsing Data ---")
        try:
            with open(self.input_filepath, 'r') as f:
                raw_data = json.load(f)
            
            parsed_records = []
            for record in raw_data:
                action_data = record.get('actionData', {})
                amount_str = action_data.get('amount')
                price_str = action_data.get('assetPriceUSD')
                
                try:
                    amount = Decimal(amount_str) if amount_str is not None else Decimal(0)
                    price = Decimal(price_str['$numberDecimal'] if isinstance(price_str, dict) else price_str)
                    amount_usd = float(amount * price)
                except:
                    amount_usd = 0.0

                parsed_records.append({
                    'userAddress': record.get('userWallet'),
                    'timestamp': pd.to_datetime(record.get('timestamp'), unit='s'),
                    'action': record.get('action', 'unknown').lower(),
                    'amountUSD': amount_usd
                })

            self.df = pd.DataFrame(parsed_records)
            print(f"✅ Successfully loaded and parsed {len(self.df)} transactions.")
            return True
        except FileNotFoundError:
            print(f"❌ ERROR: File not found at '{self.input_filepath}'")
            return False
        except Exception as e:
            print(f"❌ ERROR: An error occurred while loading data: {e}")
            return False

    def engineer_features(self):
        print("\n--- Step 2: Engineering Features ---")
        agg_features = self.df.groupby('userAddress').agg(
            first_tx=('timestamp', 'min'),
            last_tx=('timestamp', 'max'),
        )
        
        action_features = self.df.pivot_table(
            index='userAddress',
            columns='action',
            values='amountUSD',
            aggfunc=['sum', 'count'],
            fill_value=0
        )
        action_features.columns = ['_'.join(map(str, col)) for col in action_features.columns]
        wallets = agg_features.join(action_features, how='left').fillna(0)
        
        wallets['account_age_days'] = (wallets['last_tx'] - wallets['first_tx']).dt.days
        wallets['liquidation_count'] = wallets.get('count_liquidationcall', 0)
        
        total_borrowed = wallets.get('sum_borrow', 0)
        total_repaid = wallets.get('sum_repay', 0)
        total_deposited = wallets.get('sum_deposit', 0)

        wallets['repayment_ratio'] = total_repaid / (total_borrowed + 1)
        wallets['ltv_proxy'] = total_borrowed / (total_deposited + 1)
        self.wallet_features = wallets
        print("✅ Features engineered for all wallets.")
        
    def calculate_scores(self):
        print("\n--- Step 3: Calculating Scores ---")
        score = pd.Series(500.0, index=self.wallet_features.index)
        score += self.wallet_features['account_age_days'] * 0.1
        score += np.log1p(self.wallet_features.get('sum_deposit', 0) + self.wallet_features.get('sum_borrow', 0)) * 5
        score += (self.wallet_features['repayment_ratio'] - 1).clip(upper=1) * 100
        score -= self.wallet_features['liquidation_count'] * 250
        score -= self.wallet_features['ltv_proxy'].clip(upper=2) * 150

        scaler = MinMaxScaler(feature_range=(0, 1000))
        final_scores = scaler.fit_transform(score.values.reshape(-1, 1)).astype(int)
        self.wallet_scores = pd.DataFrame(final_scores, index=self.wallet_features.index, columns=['credit_score'])
        print("✅ Credit scores calculated and scaled.")

    def save_results(self):
        print("\n--- Step 4: Saving Results ---")
        scores_filepath = os.path.join(self.output_dir, 'wallet_scores.csv')
        self.wallet_scores.to_csv(scores_filepath)
        print(f"✅ Wallet scores saved to '{scores_filepath}'")

        plt.figure(figsize=(10, 6))
        plt.hist(self.wallet_scores['credit_score'], bins=range(0, 1001, 100), edgecolor='black')
        plt.title('Wallet Score Distribution')
        plt.xlabel('Credit Score Range')
        plt.ylabel('Number of Wallets')
        plt.xticks(range(0, 1001, 100))
        plt.grid(axis='y', alpha=0.75)
        
        plot_filepath = os.path.join(self.output_dir, 'score_distribution.png')
        plt.savefig(plot_filepath)
        print(f"✅ Score distribution plot saved to '{plot_filepath}'")

    def run(self):
        if self.load_and_parse_data():
            self.engineer_features()
            self.calculate_scores()
            self.save_results()
            print("\n🎉 Pipeline finished successfully!")