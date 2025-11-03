# 🧠 Hangman AI Agent — HMM + Reinforcement Learning

> **Team Project for AI / Machine Learning**
>
> An intelligent Hangman-playing agent built using **Hidden Markov Models (HMM)** for language understanding and **Reinforcement Learning (RL)** for strategic guessing.

---

## 👥 Team Members


 Kushal Nayak [https://github.com/kushalnayakm)  
 
 Abhay H Bhargav  [https://github.com/Abhay-Bhargav) 
 
 Member 3 Name   [@member3-id](https://github.com/member3-id) 
 
 Member 4 Name   [@member4-id](https://github.com/member4-id) 

---

## 📘 Project Overview

This project develops a self-learning Hangman AI system that can intelligently guess hidden words.  
It combines:
- 🧩 **HMM** to learn English letter transition probabilities from a word corpus.  
- 🤖 **RL Agent (Q-Learning / DQN)** to decide which letters to guess, based on rewards and penalties.

The agent improves over thousands of training games, learning strategies that maximize its total reward and win rate.

---

## 🧩 System Components

| File | Description |
|------|--------------|
| **hmm_model.py** | Builds and trains a Hidden Markov Model to predict letter probabilities. |
| **hangman_env.py** | Simulates the Hangman game (lives, guesses, rewards). |
| **RL_agent.py** | Contains the Reinforcement Learning agent, trainer, and evaluator classes. |
| **train_agent.py** | Connects HMM, environment, and RL agent to train the model. |
| **evaluate.py** | Tests the trained agent and reports performance metrics. |
| **cleaned_corpus.txt** | Cleaned English word list used to train the HMM model. |
| **train_words.txt / test_words.txt** | Training and testing splits of the dataset. |

---

## ⚙️ Technologies Used

- Python 3  
- NumPy  
- PyTorch  
- Matplotlib  
- Google Colab / Jupyter  
- Git & GitHub for version control
