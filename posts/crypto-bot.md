# Building My First Crypto Trading Bot: A Journey with o1-pro 

*By Denizcan Billor*

## Introduction

When I first set out to build a crypto trading bot, I never expected it to become such a multifaceted learning experience. Armed with curiosity, a bit of capital, and the patient guidance of ChatGPT, I ended up creating a momentum-based trading bot that interacts directly with on-chain liquidity pools. Along the way, I learned about the complexities of Ethereum transactions, the importance of good parameter choices, and the reality of network fees. In this post, I’ll walk you through why we chose a momentum strategy, what challenges we faced, and share some snippets of the code from the project.

If you want to dive into the full code, check out my repository here:  
[**GitHub: dbillor/trading_bot**](https://github.com/dbillor/trading_bot)

## Why a Momentum Strategy?

I’m not a pro trader by any means, so I needed a strategy that was straightforward and helped me focus on core mechanics rather than overly complex analytics. Momentum trading met that criterion nicely:

- **Simple Rules:** If the token’s price rises by a certain percentage within a defined window, that’s our buy signal. If it falls by a certain threshold, that’s our sell signal.
- **Clear Thresholds:** By setting a `momentum_threshold_percent` in our config, we keep it explicit and easy to tweak.
- **No Need for Deep Models:** We didn’t have to implement machine learning or advanced TA indicators on day one. This let me learn the ropes of fetching data, signing transactions, and executing swaps first.

## The Challenges We Faced

### Gas Fees & Mainnet Reality

One of the biggest eye-openers was the cost of doing any meaningful operation on Ethereum mainnet. Swapping $30 worth of WETH to USDC might cost $4 or even more, depending on congestion. This made testing small trades financially daunting.

**What we considered:**
- Using Layer-2 solutions like Arbitrum or Optimism, where fees are much lower.
- Adjusting trade sizes and frequency to reduce costs per trade.

### Balances & Trade Size

At first, I tried using a `base_position_size_wei` that was way too large for my modest starting capital. I kept hitting “Insufficient token_in balance” errors.

**Lesson learned:**  
Match your `base_position_size_wei` to your actual wallet balance and the current ETH price. Starting small and scaling up once you know it works is essential.

### On-Chain Details: Approvals, Nonces, and Timeouts

Swapping tokens isn’t just a function call; you must approve the Uniswap router to spend your tokens, handle transaction nonces, and consider what happens if a transaction is stuck.

**Adjustments we made:**
- Added a function to `approve_token()` before trading.
- Included retry logic and dynamic fee adjustments for pending transactions.
- Ensured we fetched the latest nonce with `get_transaction_count(self.address, 'pending')` to avoid nonce clashes.

## Code Snippets That Tell the Story

Below are a few snippets from the repo that highlight key parts of the application. For the full context, check out the [**GitHub repository**](https://github.com/dbillor/trading_bot).

### Momentum Logic (from `momentum_logic.py`)
This snippet shows how we decide when to buy or sell based on recent price changes:
```python
def should_buy(self):
    if len(self.prices) < 2:
        return False
    oldest_price = self.prices[0][1]
    latest_price = self.prices[-1][1]
    increase = (latest_price - oldest_price) / oldest_price * 100.0
    return increase >= self.threshold_percent

def should_sell(self):
    if len(self.prices) < 2:
        return False
    oldest_price = self.prices[0][1]
    latest_price = self.prices[-1][1]
    decrease = (oldest_price - latest_price) / oldest_price * 100.0
    return decrease >= (self.threshold_percent / 2)

```

**Why it matters:**  
This logic is simple, but effective as a starting point. The code’s clarity makes it easy to tweak parameters and thresholds.

### Approving Tokens for Trades (from `trade_executor.py`)

Before swapping tokens, the router must be approved:

```python
def approve_token(self, token_address, spender_address, amount_in_wei):
    token_contract = self.w3.eth.contract(address=Web3.to_checksum_address(token_address), abi=self.erc20_abi)
    nonce = self.w3.eth.get_transaction_count(self.address, 'pending')

    tx = token_contract.functions.approve(spender_address, amount_in_wei).buildTransaction({
        'from': self.address,
        'gas': 100000,
        'gasPrice': self.w3.eth.gas_price,
        'nonce': nonce,
        'chainId': 1
    })

    signed_tx = self.account.sign_transaction(tx)
    tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.status == 1

```

**Why it matters:**  
No approval, no trade. This snippet shows the raw, under-the-hood steps of creating a transaction, signing it, and waiting for a receipt.

### Executing a Buy (also from `trade_executor.py`)

Here’s the core of swapping tokens once we have a signal:

```python
def execute_buy(self, token_in, token_out, amount_in_wei, token_in_price):
    balance = self.get_token_balance(token_in)
    if balance < amount_in_wei:
        self.logger.error("Insufficient token_in balance to execute buy.")
        return None

    amount_out_min = self.get_amount_out_min(amount_in_wei, token_in_price)
    nonce = self.w3.eth.get_transaction_count(self.address, 'pending')
    gas_price = self.w3.eth.gas_price

    tx = self.build_exact_input_single(token_in, token_out, 3000, self.address, amount_in_wei, amount_out_min)
    built_tx = tx.buildTransaction({
        'from': self.address,
        'gas': 300000,
        'gasPrice': gas_price,
        'nonce': nonce,
        'chainId': 1,
        'value': 0
    })

    signed_tx = self.account.sign_transaction(built_tx)
    tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.status == 1

```

**Why it matters:**  
This code encapsulates the entire process: checking balances, calculating minimum outputs for slippage protection, sending the transaction, and confirming it.

## Conclusion

Working through the process of building this bot—setting up a momentum strategy, dealing with high gas fees, adjusting trade sizes, and mastering on-chain interactions—has been an incredible learning experience. It’s shown me that even a simple strategy can quickly get complex once we involve actual blockchain mechanics.

The code and snippets shared here represent just a glimpse of the project. For the full picture, I invite you to explore the [**GitHub repository**](https://github.com/dbillor/trading_bot) and maybe even try it out on a testnet or a Layer-2 environment. With a bit more tuning and cheaper fees, this approach could form the foundation of a more refined, profitable trading system.

----------

If you find this journey inspiring or educational, I encourage you to experiment. Start small, adjust parameters, and don’t be discouraged by the intricacies of on-chain trading. Over time, you’ll gain the insights and confidence needed to tackle more complex strategies and ecosystems.
