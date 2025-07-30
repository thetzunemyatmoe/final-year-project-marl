# Solving Energy Efficient Multi-UAV Coverage Control Problem by Multi-Agent Reinforcement Learning Based Approach

## Description
This project addresses the coverage control problem for multiple unmanned aerial vehicles (UAVs) using a multi-agent reinforcement learning (MARL) approach. The problem is formulated within a game-theoretic framework and solved using an Independent Advantage Actor with Centralized Critic (IA2CC) algorithm. Validation and evaluation include hyperparameter tuning, reward function analysis, and generalization testing of the trained models.

## Table of Content
1. Problem Formulation
2. Enviroment and Algorithm Implementation
3. Validation and Evaluation Results


## 1. Problem Formulation

The problem is formulated using the **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** framework.  
Dec-POMDP is an extension of the standard Markov Decision Process (MDP) to multi-agent scenarios and offers:

- **Partial Observability**  
  Each UAV operates with limited local information about the environment.  

- **Decentralized Decision-Making**  
  UAVs make decisions independently, without relying on centralized control.  

- **Cooperative Behavior**  
  A shared global reward function trains agents to coordinate and complete tasks collectively.  

This formulation closely reflects real-world scenarios, such as a team of UAVs performing area coverage with only local observations and no direct inter-agent communication.

## 2. Environment and Algorithm Implementation

### Environment

The file **`IA2CC/environment.py`** models the environment dynamics and simulates interactions among multiple UAVs within a **discrete 2D grid world**.  
It also handles **reward calculation** and adheres to the **OpenAI Gym API** design principles, ensuring compatibility and ease of experimentation.

Key features include:

- **`env.reset()`**  
  Resets the environment to its initial state, enabling a fresh start for each training episode.

- **`env.step(actions)`**  
  Executes a single step in the environment based on the **collection of actions** chosen by all UAVs, returning the next state, reward, and other relevant information.

This modular design facilitates smooth integration with reinforcement learning algorithms and accelerates development and testing.


### Algorithm

The **IA2CC** algorithm is an extension of the **Advantage Actor-Critic (A2C)** framework, designed specifically for **multi-agent scenarios**.  
It adopts the widely used **Centralized Training with Decentralized Execution (CTDE)** paradigm, where agents are trained with access to global information but operate independently using only local observations during execution.


The file **`IA2CC/IA2CC.py`**,  **`IA2CC/Actor.py`**,  **`IA2CC/Critic.py`** implements the core components of the algorithm.

- **`Actor` Class (`Actor.py`)**  
  Defines the **actor network** responsible for policy learning.  
  Implemented using **PyTorch**, it is a **feedforward neural network** consisting of three fully connected layers,with **ReLU** activation functions applied to each hidden layer to introduce non-linearity and improve learning performance.

- **`Critic` Class (`Critic.py`)**  
  Defines the **centralized critic network** used for value estimation.  
  Similarly implemented in **PyTorch**, it is a **feedforward neural network** with three fully connected layers,each hidden layer utilizing **ReLU** activation functions for effective representation learning.

- **`IA2CC` Class (`IA2CC.py`)**  
  Encapsulates the **model initialization**, including both the **actor** and **critic** networks.  
For flexibility and fine-tuning, it accepts configurable parameters such as:

  - Actor input and output sizes  
  - Critic input size  
  - Number of agents  
  - Learning rates  
  - Discount factor  
  - Entropy weight  

  This class is carefully designed with **scalability, modularity, and reusability** in mind,  
making it adaptable for various multi-agent reinforcement learning setups.  


- **`IA2CC/learner.py`**  
  Contains the **`train()`** function, which implements the **step-by-step logic of the training loop**, including interaction with the environment, experience collection, loss computation, and network updates.

## 3. Validation and Evaluation Results

### Hyperparameter Tuning

The following parameters were tuned during training:  
1. Learning Rate  
2. Entropy Coefficient  
3. Discount Factor  

#### Learning Rate

In IA2CC, the actor and critic networks each have their own independent learning rates.  
Six different learning rate configurations were evaluated over **5000 training episodes**, each consisting of **100 steps**:

| Configuration # | Actor Learning Rate | Critic Learning Rate |
|-----------------|--------------------|---------------------|
| 1               | 0.0001             | 0.0001              |
| 2               | 0.0001             | 0.0003              |
| 3               | 0.0003             | 0.0001              |
| 4               | 0.00001            | 0.0001              |
| 5               | 0.0001             | 0.001               |
| 6               | 0.00003            | 0.0003              |

![Reward Trend](IA2CC/reward/learningrate/reward_trend_learningrate.png)
**Figure 1:** Reward trend during training under different learning rate configurations,showing how varying actor and critic learning rates impact convergence and performance.


