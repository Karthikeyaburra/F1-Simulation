# 🏎️ F1 Race Strategy Optimizer
> **Monte Carlo Simulation Engine & Reinforcement Learning Dashboard**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![FastF1](https://img.shields.io/badge/FastF1-3.1+-e10600?style=for-the-badge&logo=formula1&logoColor=white)](https://docs.fastf1.dev/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-000000?style=for-the-badge)](https://gymnasium.farama.org/)

---

## 📌 Project Overview

An end-to-end full-stack web application and simulation suite designed to test custom Formula 1 race strategies and predict finishing position probabilities in real time.

Built on the peer-reviewed methodology by **Heilmeier et al. (2020)** (*"Application of a Monte Carlo Simulation for Race Strategy Optimization in Formula 1"*), the engine executes **500+ stochastic race simulations per session** — modelling driver pace, tyre degradation state spaces, pit stop crew variability, Bayesian DNF probabilities, and Safety Car / VSC phases.

Furthermore, the simulator is structured as a **custom Gymnasium-compatible Reinforcement Learning (RL) environment** (`F1StrategyEnv`) trained via Tabular Q-Learning to autonomously determine optimal pit-stop timing and compound selection under uncertainty.

---

## ✨ Key Features

### 🎲 Monte Carlo Race Simulation Engine
* **Lap-by-Lap Stochastic Progression**: Simulates complete race distances (50–66 laps) across multiple drivers under continuous lap time variability.
* **State-Space Tyre Degradation**: Models tyre wear ($\alpha_t$) with process noise ($\sigma_\eta$) and asymmetric observation noise ($\sigma_\epsilon$, skewness $\alpha$).
* **Fisk (Log-Logistic) Pit Stop Variability**: Models team-specific pit stop durations using fitted Fisk distributions ($\mathcal{F}(c, \text{loc}, \text{scale})$).
* **Bayesian Beta-Bernoulli DNF Model**: Estimates driver accident ($P_{\text{accident}}$) and team failure ($P_{\text{failure}}$) probabilities using Beta priors updated per track type (*street*, *permanent*, *high-speed*).
* **Full Course Yellow (FCY) Pre-Determination**: Simulates Safety Car (SC) and Virtual Safety Car (VSC) frequency, deployment timing, and duration from historical data.

### 🤖 Reinforcement Learning Agent (`F1StrategyEnv`)
* **Gymnasium Interface**: Implements `reset()`, `step()`, `observation_space`, and `action_space`.
* **State Space ($12\text{D}$ continuous)**: Race progress, tyre age, compound (one-hot), fuel mass, position, gap-to-leader, SC status, latent wear state, pit count.
* **Action Space ($4$ discrete)**: `0: CONTINUE`, `1: PIT (SOFT)`, `2: PIT (MEDIUM)`, `3: PIT (HARD)`.
* **Reward Structure**: F1 points system (P1=25, P2=18... P10=1) + position-gain shaping rewards + compound diversity rule penalties.

### 📊 Production Analytics Dashboard (React + WebSockets)
* **Position Probability Heatmap**: $20 \times 20$ interactive grid representing $P(\text{Driver}_i, \text{Position}_j)$.
* **WebSocket Progress Streaming**: Live progress bar streaming cumulative probabilities via `/ws/simulate`.
* **Visual Stint Timeline & Strategy Editor**: Drag-and-drop / slider strategy editor with live compound stint bars.
* **RL Strategy Comparison Panel**: Side-by-side comparison between human user strategy and RL-recommended strategy with position ($\Delta \text{Pos}$) and points ($\Delta \text{Pts}$) deltas.
* **Multi-GP Support**: Built-in configurations for 8 circuits (*Bahrain, Saudi Arabia, Australia, Japan, Miami, Spain, Great Britain, Italy*).

---

## 📐 Mathematical Formulation

### Core Lap Time Equation (Heilmeier et al., Eq. 1)

$$t_{\text{lap}}(l) = t_{\text{base}} + t_{\text{car}} + t_{\text{driver}} + t_{\text{fuel}}(l) + t_{\text{tyre}}(a, c) + t_{\text{grid}}(l) + t_{\text{pit}}(l) + t_{\text{var}}$$

Where:
* **Fuel Effect**: $t_{\text{fuel}}(l) = \gamma_{\text{mass}} \cdot (m_{\text{fuel}}(l) - m_{\text{mid}})$
* **State-Space Tyre Wear**: $\alpha_{t} = \alpha_{t-1} + \nu_c + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)$
* **Lap Time Observation Noise**: $t_{\text{var}} \sim \text{SkewNorm}(\alpha, 0, \sigma_\epsilon)$
* **Pit Duration Offset**: $t_{\text{pit\_var}} \sim \text{Fisk}(c, \text{loc}, \text{scale})$
* **Bayesian DNF Update**: $P(\text{Event} \mid z, N) = \frac{\hat{\alpha} + z}{\hat{\alpha} + \hat{\beta} + N}$

---

## 🏗️ System Architecture

```
                                    +-----------------------------------+
                                    |         React 18 Dashboard        |
                                    |   Vite + Framer Motion + Recharts |
                                    +-----------------+-----------------+
                                                      |
                                          HTTP REST & WebSocket (/ws)
                                                      |
                                    +-----------------v-----------------+
                                    |          FastAPI Backend          |
                                    |        (app.py / Uvicorn)         |
                                    +--------+-----------------+--------+
                                             |                 |
                       +---------------------+                 +---------------------+
                       |                                                             |
            +----------v----------+                                       +----------v----------+
            |  Monte Carlo Engine |                                       |    RL Environment   |
            | (race_simulator.py) |                                       | (rl_environment.py) |
            +----------+----------+                                       +----------+----------+
                       |                                                             |
                       +---------------------+                 +---------------------+
                                             |                 |
                                    +--------v-----------------v--------+
                                    |        Statistical Models         |
                                    |  FastF1 Telemetry Data Pipeline   |
                                    |  Fisk, SkewNorm, Bayesian Beta    |
                                    +-----------------------------------+
```

---

## ⚡ Quick Start

### 1. Clone & Setup Repository
```bash
git clone https://github.com/YOUR_USERNAME/F1-Simulation.git
cd F1-Simulation
```

### 2. Install Python Dependencies
```bash
# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install required packages
pip install -r requirements.txt
```

### 3. Launch Backend API
```bash
cd src
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
*API documentation will be available at `http://localhost:8000/docs`.*

### 4. Launch React Frontend
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
*Web Dashboard will open at `http://localhost:3000`.*

---

## 📡 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/tracks` | Returns supported GP circuits and lap parameters |
| `GET` | `/api/config?year=2024&gp_name=Bahrain` | Baseline race entry list and team pace offsets |
| `GET` | `/api/analytics?year=2024` | Fitted statistical distribution parameters |
| `POST` | `/api/simulate` | Executes synchronous batch Monte Carlo simulation |
| `WS` | `/ws/simulate` | WebSocket endpoint streaming live simulation progress |
| `POST` | `/api/simulate/rl` | Trains Q-learning RL agent & returns strategy comparison |

---

## 🧪 Model Validation Metrics

Evaluated against actual Formula 1 Grand Prix results:

| Metric | Target / Result |
| :--- | :--- |
| **Position MAE** | $\approx 3.2$ finishing positions |
| **Top-10 Accuracy** | $\approx 70\%$ precision/recall |
| **Winner Top-3 Coverage** | $78\%$ across validation races |
| **Simulation Speed** | $> 500$ iterations in $< 3.5$ seconds |

---

## 📄 Citation

```bibtex
@article{heilmeier2020application,
  title={Application of a Monte Carlo Simulation for Race Strategy Optimization in Formula 1},
  author={Heilmeier, Alexander and Graf, Markus and Betz, Johannes and Lienkamp, Markus},
  journal={Applied Sciences},
  volume={10},
  number={12},
  pages={4229},
  year={2020},
  publisher={MDPI}
}
```

---

## 📝 License

Distributed under the MIT License. Data sourced via [FastF1](https://docs.fastf1.dev/).
