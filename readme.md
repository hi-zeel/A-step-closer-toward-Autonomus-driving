# Reinforcement Learning for Self-Driving Cars 🚗🤖

## 📌 Project Overview
This project implements **Reinforcement Learning (RL)** to train a simulated autonomous car using a Deep Q-Network (DQN).  
The goal is to learn safe driving strategies (accelerating, braking, lane changes) in a multi-lane environment with surrounding vehicles.

The implementation is inspired by MIT’s **DeepTraffic** benchmark but extended for custom experimentation.

---

## 🎯 Key Learning Outcomes
By working on this project, I practiced:
- Designing and training **Deep Q-Learning (DQN)** agents.  
- Implementing **experience replay** and **target networks** for stability.  
- Building and training a **Convolutional Neural Network (CNN)** for state representation.  
- Logging and visualizing training progress (rewards, Q-values).  
- Running experiments with different hyperparameters (learning rate, batch size, network depth).  

---

## 🗂️ Repository Structure

- **`gui.py`** → Main simulation loop with Pygame visualization (car, road, traffic, overlays).  
- **`deep_traffic_agent.py`** → Defines the **RL agent** (DQN + epsilon-greedy exploration).  
- **`cnn.py`** → Convolutional Neural Network used to approximate Q-values.  
- **`config.py`** → Central place for hyperparameters (learning rate, episodes, batch size, etc.).  
- **`log.py`** → Logging system for rewards and Q-values (writes `.log` files).  
- **`results.py`** → Script to read logs and plot results.  
- **`car.py`** → Car dynamics, decision-making, and interaction with agent.  
- **`gui_util.py`** → Helper functions for visualization (road, cars, stats).  
- **`advanced_view/`** → Extra visualization features.  
- **`log/`** → Training logs (reward.log, q_values.log).  

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/hi-zeel/A-step-closer-toward-Autonomus-driving.git
cd A-step-closer-toward-Autonomus-driving
```

### 2. Install dependencies
```bash
conda create -n rlcar python=3.7
conda activate rlcar
pip install -r requirements.txt
```

### 3. Start training
```bash
python gui.py
```

This launches the simulation and starts training the DQN agent.  
- During training, logs are written inside `log/round*/…/`.  
- You can watch the car drive in real-time if `VISUALENABLED=True` in `config.py`.  

### 4. Plot results
```bash
python results.py
```

This reads the `reward.log` and `q_values.log` files and produces reward curves.

---

## 📊 Example Output
- **Rewards curve** showing how average rewards evolve during training.  
- **Q-values statistics** (min, mean, max, std) to analyze learning stability.  

---

## 🔮 Next Steps
- Add **Double DQN / Dueling DQN** for better stability.  
- Experiment with **continuous control (DDPG, PPO)**.  
- Integrate with **CARLA Simulator** for more realistic driving.  

---

✨ This project helped me understand the **full pipeline of deep reinforcement learning for autonomous driving**, from environment setup to training, logging, and evaluation.
