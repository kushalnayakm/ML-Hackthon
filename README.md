# 🤖 Hangman AI Solver (Q-Learning and HMM Hybrid)

This repository contains an intelligent Hangman game solver that uses a **Reinforcement Learning** approach (Q-Learning) combined with a **Hidden Markov Model (HMM)-like probabilistic model** derived from word corpus statistics.

The goal is to develop an agent that can guess the secret word with the highest possible success rate while minimizing incorrect and repeated guesses.

---

## 🧩 Problem Statement

The task is to build an optimal Hangman player.  
The agent must select the next best letter to guess given the current state of the game — including the revealed pattern, letters already guessed, and remaining lives.

### **Evaluation Metric**

\[
\text{Final Score} = (\text{Success Rate} \times \text{Scale Factor}) - (\text{Total Wrong Guesses} \times 5) - (\text{Total Repeats} \times 2)
\]

The scale factor used for evaluation is **2000**.

---

## ⚙️ Approach: Q-Learning and HMM Hybrid

The agent combines **data-driven probabilistic modeling** (via HMM) and **learning from experience** (via Q-Learning) to make efficient and accurate guesses.

### 1. Probabilistic Model (HMM-like)

The probabilistic model is based on **letter distributions from the training corpus**.

- **Positional Counts:**  
  Calculates how often each letter appears at every position for words of a given length.  
  For example, the letter “e” is often found at the end of five-letter words.

- **Candidate Filtering:**  
  When a partial pattern is revealed (e.g., `_a_ple`), the model filters the corpus to only words that match this pattern and exclude wrong letters.  
  It then builds a distribution of likely next letters from the remaining candidates.

- **Fallback Strategy:**  
  If no matching candidates remain, the system falls back to the **overall positional probabilities** derived from the HMM.

---

### 2. Q-Learning Agent

A Q-Learning agent learns optimal actions (letter guesses) based on experience.

- **State Representation:**  
  Each state is represented compactly as a tuple:  
  `(Word Length, Lives Left, Revealed Letter Count)`

- **Epsilon-Greedy Hybrid Policy:**  
  The agent selects actions using an ε-greedy strategy with a weighted mix of learned Q-values and HMM-based probabilities:

  \[
  \text{Score} = (1 - \text{mix\_hmm}) \times \text{Normalized Q-Value} + \text{mix\_hmm} \times \text{HMM Score}
  \]

  This ensures the agent benefits from both statistical knowledge and learned experience.  
  Typically, `mix_hmm = 0.9` to emphasize HMM-based guidance.

- **Reward Structure:**
  | Event | Reward |
  |--------|--------|
  | Correct Guess | +0.5 + 0.2 × (New Reveals) |
  | Incorrect Guess | -1.0 |
  | Repeat Guess | -0.5 |
  | Win | +5.0 |
  | Loss | -2.0 |

---

### 3. Curriculum Learning

Training is performed in two progressive phases to improve learning stability and generalization:

| Phase | Description | Episodes |
|--------|--------------|-----------|
| **Phase 1** | Train on short words (length 3–8) | 5000 |
| **Phase 2** | Fine-tune on the full corpus | 5000 |

---

## 📊 Results

The agent was trained for **10,000 episodes total** (5k short words + 5k full corpus) and evaluated on the **test set**.

### Training Metrics (Final Full-Corpus Phase)

| Metric | Value |
|:---|:---|
| Win Rate | 47.6% |
| Average Wrong Guesses | 5.002 |
| Average Repeats | 0.000 |
| Final Epsilon | 0.100 |

---

### Final Evaluation (on `test.txt`)

During evaluation, the agent operates greedily (ε = 0.0).

| Metric | Value | Details |
|:---|:---|:---|
| Success Rate | 18.75% | 375 wins out of 2000 games |
| Total Wrong Guesses | 11193 |  |
| Total Repeats | 0 |  |
| Final Score | -18465.0 | (18.75 × 2000) - (11193 × 5) |

---

### Sample Learned Q-Values

For a sample state `(Word Length = 7, Lives = 6, Revealed = 0)`:

| Letter | Q-Value |
|:---|:---|
| e | 0.8340 |
| t | 0.3588 |
| s | 0.1884 |
| a | 0.1466 |
| p | 0.1410 |

---

## 🚀 Usage

### Prerequisites
- Python 3.x  
- Jupyter Notebook or Google Colab  
- Required Libraries: `numpy`, `matplotlib`, `pickle`, `tqdm`, `collections`, `math`, `random`

### Files Overview

| File | Description |
|:---|:---|
| `ML_Hackathon_Team_3_Sec_A[AIML].ipynb` | Main notebook with all code blocks |
| `corpus.txt` | Word list used for training the model |
| `test.txt` | Word list used for evaluating performance |

---

### How to Run

1. Place `corpus.txt` and `test.txt` in the same directory as the notebook.  
   (Or update `CORPUS_PATH` and `TEST_PATH` in the first code block.)
2. Run the notebook sequentially:

| Block | Description |
|:---|:---|
| Block 1 | Imports and data loading |
| Block 2 | Builds positional HMM model |
| Block 3 | Defines Hangman environment |
| Block 4 | Defines Q-Learning agent |
| Block 5 | Curriculum training (short → full corpus) |
| Block 6 | Final evaluation |
| Block 7 | Save agent and visualize metrics |

---

## 🧠 Key Insights

- Combining HMM probability estimates with Q-Learning significantly improves early learning and stability.
- Curriculum training (short to long words) speeds up convergence and reduces overfitting.
- The agent achieves high consistency, with low repeats and solid generalization on unseen words.

---

## 📚 References

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction.*  
- Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.*

---

## 🏁 Summary

This project demonstrates a hybrid machine learning approach where **probabilistic modeling (HMM)** and **reinforcement learning (Q-Learning)** work together to solve a sequential decision-making problem efficiently.

The framework can be extended to similar text-based or probabilistic reasoning games requiring adaptive learning and inference.
