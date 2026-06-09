# 🚁 Drone-Falco Reinforcement Learning

A complete Reinforcement Learning (RL) framework for training an autonomous quadcopter controller using **Proximal Policy Optimization (PPO)**. The project features a custom drone simulation built with **PyBullet**, a **PyTorch**-based PPO implementation, and a comprehensive evaluation suite with an interactive KPI dashboard.

---

## ✨ Features

### 🚁 Custom Drone Simulation

* OpenAI Gym-compatible environment built with **PyBullet**.
* Realistic quadcopter dynamics and physics simulation.
* Custom drone model rendered from URDF and OBJ files.

### 🧠 PPO Reinforcement Learning

* Actor-Critic architecture implemented in **PyTorch**.
* Generalized Advantage Estimation (GAE).
* KL-divergence early stopping.
* Learning-rate annealing.
* Automatic checkpoint saving.
* Real-time training progress GUI built with **Tkinter**.

### 📊 Advanced Performance Evaluation

* Automated policy testing across multiple episodes.
* Control-theory KPIs including:

  * Settling Time
  * Steady-State Error
  * Overshoot
  * Hover Time
  * Control Effort
  * Maximum Tilt

### 📈 Interactive Dashboard

* Upload evaluation results directly in your browser.
* Dynamic charts and performance summaries.
* No additional backend required.

---

## 📂 Project Structure

```text
Drone-Falco-Reinforcement-Learning/
│
├── RL_code_quadcopt.py            # PPO training script
├── quad_env.py                    # Custom PyBullet environment
├── eval_quad.py                   # Policy evaluation and KPI generation
├── quadcopter_kpi_dashboard.html  # Interactive KPI dashboard
├── quadrotor.urdf                 # Drone model definition
├── quadrotor_base.obj             # 3D mesh for visualization
├── requirements.txt               # Project dependencies
└── README.md
```

### Main Files

| File                            | Description                                                               |
| ------------------------------- | ------------------------------------------------------------------------- |
| `RL_code_quadcopt.py`           | Main PPO training pipeline and learning loop.                             |
| `quad_env.py`                   | Drone dynamics, reward function, observations, and simulation management. |
| `eval_quad.py`                  | Evaluates trained models and exports KPI metrics to JSON.                 |
| `quadcopter_kpi_dashboard.html` | Visualizes evaluation results through interactive charts.                 |

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Matteo7100/Drone-Falco-Reinforcement-Learning.git
cd Drone-Falco-Reinforcement-Learning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Linux Users

If you encounter issues related to **Tkinter**, install it through your package manager:

```bash
sudo apt-get install python3-tk
```

---

## ⚙️ Configuration

Before running the simulation, update the URDF path in `quad_env.py` to match your local setup.

Locate the following line:

```python
self.drone = p.loadURDF(r"C:\Path\To\Your\Drone_Falco\quadrotor.urdf")
```

Replace it with the correct path to your `quadrotor.urdf` file.

> **Tip:** Using a relative path instead of an absolute path will make the project more portable across different machines.

---

## 🎮 Usage

### Train the Agent

Start PPO training:

```bash
python RL_code_quadcopt.py
```

During training:

* A progress window will display training status.
* Checkpoints are automatically saved.
* The best-performing model is stored as:

```text
quad_model_best.pth
```

---

### Evaluate a Trained Policy

Run evaluation over multiple episodes:

```bash
python eval_quad.py --model quad_model_best.pth --episodes 20
```

This generates:

```text
eval_results.json
```

containing performance metrics and episode statistics.

---

### Visualize Evaluation Results

Open:

```text
quadcopter_kpi_dashboard.html
```

in any modern web browser and drag-and-drop the generated `eval_results.json` file.

The dashboard provides:

* Success, Flip, and Out-of-Bounds rates
* Mean Distance-to-Target over time
* Reward distribution
* Hover Time analysis
* Control Effort metrics
* Maximum Tilt measurements

---

## 📊 Example Workflow

```text
Train PPO Agent
        ↓
Save Best Model
        ↓
Run Evaluation
        ↓
Generate eval_results.json
        ↓
Load Dashboard
        ↓
Analyze KPIs
```

---

## 🤝 Contributing

Contributions are welcome and greatly appreciated.

To contribute:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature/AmazingFeature
```

3. Commit your changes:

```bash
git commit -m "Add AmazingFeature"
```

4. Push to your branch:

```bash
git push origin feature/AmazingFeature
```

5. Open a Pull Request.

Bug fixes, reward-function improvements, physics enhancements, and dashboard features are all encouraged.

---

## 👤 Author

**Matteo7100**

GitHub: https://github.com/Matteo7100

---

## 📜 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
