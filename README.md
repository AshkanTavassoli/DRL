# This is a modification of a code proposed in the paper below trying to achieve a better score - Our improvements:
## Fund allocation update
We found that original code does not purchase stock as smart as it can be. The authors of this paper chose to start buying from the smallest request to biggest ones in each step – smallest means the smallest number of shares asked for and not the amount of money each request actually costs. The problem here is the worst-case scenario where there is not enough money left to perform the big buy requests in the end – which happens a lot based on our tests. It is not a good strategy since we expect the Agent to make a big buy request when it believes there is a great reward in return and make small buy requests when the rewards are expected to be smaller than the others (Expected Q Value), So by just buying the small buy requests in a lot of steps we are choosing the path that will lead to smaller expected reward.
There is a problem with the training process as well, Sometimes the agent asks for an action and goes to a new state and trains itself using the actual outcome of that action but what happens is that only a part of the requested action has taken place and the reward that the agent is trying to train on is not from the action it requested. In actor critic methods this definitely would misguide the critic because it will be trained on a dataset that in some cases the action and reward are not actually a pair and the real action is unknown to the agent. 
3.1.2 Implementation
We decided to test two other options:
●	Doing the same in reverse, meaning buying the largest buy requests coming from the RL agent at first and go on. This idea could lead to a high reward since we are always buying the stocks that we are expected to get a great reward and buy the ones with small rewards only if we have money left. But there is a downside to this, it makes the behavior much greedier since in an extreme case it could do one stock trading strategy which means a higher profit at a much higher risk. In practice this idea did not act well and we believe it is because of its poor performance in market crashes. This approach could get a much better result as long as the market is bullish.
●	Splitting the available money between all incoming buy requests while keeping the ratio between different buy requests. This approach makes sure the agent gets what it actually asks for and keeps a balance between profit and the risk it is taking by keeping a more diverse portfolio.

## Sliding window length test
The paper chooses to training and switch agents in the ensemble strategy every quarter but does not provide any reasoning. As we all know, stock markets can be volatile or stable for certain period, but it is hard to say that there is a cycle every 3 months. For example, bullish market may last longer and the market increase can be slow and steady. However, market crash can happened all of a sudden and usually does not last long. Therefore, using this 3 month window may be rigid in some cases as there could be too long or short. 
*This change did not show promising improvements
## Ensemble with 2 agents
We overserved the best selected agent might not be the best performer in the next trading quarter. Instead, the agent with second Sharpe ratio has better performance. As we can see from Jan. 2017 to Sept. 2017, the ensemble strategy changes agents but it turns out the agent with second Sharpe ratio in the current quarter performed better in the next quarter. So the strategy switches agents between PPO and A2C. However, changing agents frequently might not be an optimal solution.  
This is because the market condition varies and when the market momentum changes, bullish market can turn into bearish market quickly and vice versa.  Changing agents in a good time point is hard. In order to tackle this, we would like to utilize both agents with top 2 Sharpe ratios and with scaled portfolio value. It may lower the risk because the strategy is more diverse now with more agents working together.
*This change did not show promising improvements
## Ensemble multiple PPO models
According to the original paper, PPO has the highest return, but also the highest risk. Here we try to lower the risk of PPO by ensembling multiple PPO models and keep the high return of PPO. 

# Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy
This repository refers to the codes for [ICAIF 2020 paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996)


# Abstract
Stock trading strategies play a critical role in investment. However, it is challenging to design a profitable strategy in a complex and dynamic stock market. In this paper, we propose a deep ensemble reinforcement learning scheme that automatically learns a stock trading strategy by maximizing investment return. We train a deep reinforcement learning agent and obtain an ensemble trading strategy using the three actor-critic based algorithms: Proximal Policy Optimization (PPO), Advantage Actor Critic (A2C), and Deep Deterministic Policy Gradient (DDPG). The ensemble strategy inherits and integrates the best features of the three algorithms, thereby robustly adjusting to different market conditions. In order to avoid the large memory consumption in training networks with continuous action space, we employ a load-on-demand approach for processing very large data. We test our algorithms on the 30 Dow Jones stocks which have adequate liquidity. The performance of the trading agent with different reinforcement learning algorithms is evaluated and compared with both the Dow Jones Industrial Average index and the traditional min-variance portfolio allocation strategy. The proposed deep ensemble scheme is shown to outperform the three individual algorithms and the two baselines in terms of the risk-adjusted return measured by the Sharpe ratio.

<img src=figs/stock_trading.png width="600">

## Reference
Hongyang Yang, Xiao-Yang Liu, Shan Zhong, and Anwar Walid. 2020. Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. In ICAIF ’20: ACM International Conference on AI in Finance, Oct. 15–16, 2020, Manhattan, NY. ACM, New York, NY, USA.

## [Our Medium Blog](https://medium.com/@ai4finance/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02)
## Installation:
```shell
git clone https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020.git
```



### Prerequisites
For [OpenAI Baselines](https://github.com/openai/baselines), you'll need system packages CMake, OpenMPI and zlib. Those can be installed as follows

#### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Windows 10

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites).
    
### Create and Activate Virtual Environment (Optional but highly recommended)
cd into this repository
```bash
cd Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020
```
Under folder /Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020, create a virtual environment
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages. 

**Virtualenvs can also avoid packages conflicts.**

Create a virtualenv **venv** under folder /Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020
```bash
virtualenv -p python3 venv
```
To activate a virtualenv:
```
source venv/bin/activate
```

## Dependencies

The script has been tested running under **Python >= 3.6.0**, with the folowing packages installed:

```shell
pip install -r requirements.txt
```

### Questions

### About Tensorflow 2.0: https://github.com/hill-a/stable-baselines/issues/366

If you have questions regarding TensorFlow, note that tensorflow 2.0 is not compatible now, you may use

```bash
pip install tensorflow==1.15.4
 ```

If you have questions regarding Stable-baselines package, please refer to [Stable-baselines installation guide](https://github.com/hill-a/stable-baselines). Install the Stable Baselines package using pip:
```
pip install stable-baselines[mpi]
```

This includes an optional dependency on MPI, enabling algorithms DDPG, GAIL, PPO1 and TRPO. If you do not need these algorithms, you can install without MPI:
```
pip install stable-baselines
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more details and alternatives (from source, using docker).


## Run DRL Ensemble Strategy
```shell
python run_DRL.py
```
## Backtesting

Use Quantopian's [pyfolio package](https://github.com/quantopian/pyfolio) to do the backtesting.

[Backtesting script](backtesting.ipynb)

## Status

<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>

* 1.0.1
	Changes: added ensemble strategy
* 0.0.1
    Simple version
</div>
</details>

## Data
The stock data we use is pulled from [Compustat database via Wharton Research Data Services](https://wrds-web.wharton.upenn.edu/wrds/ds/compd/fundq).
<img src=figs/data.PNG width="500">

### Ensemble Strategy
Our purpose is to create a highly robust trading strategy. So we use an ensemble method to automatically select the best performing agent among PPO, A2C, and DDPG to trade based on the Sharpe ratio. The ensemble process is described as follows:
* __Step 1__. We use a growing window of 𝑛 months to retrain our three agents concurrently. In this paper we retrain our three agents at every 3 months.
* __Step 2__. We validate all 3 agents by using a 12-month validation- rolling window followed by the growing window we used for train- ing to pick the best performing agent which has the highest Sharpe ratio. We also adjust risk-aversion by using turbulence index in our validation stage.
* __Step 3__. After validation, we only use the best model which has the highest Sharpe ratio to predict and trade for the next quarter.

## Performance
<img src=figs/performance.png>
