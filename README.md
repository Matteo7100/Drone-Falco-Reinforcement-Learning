# Drone-Falco Reinforcement Learning

This repository contains a Reinforcement Learning (RL) project designed to train an agent to control a quadcopter (drone). The project utilizes a custom environment to simulate the physics and dynamics of the "Falco" drone using **PyBullet** and trains a **PPO (Proximal Policy Optimization)** agent using **PyTorch**.

## 📂 File Structure

* **`RL_code_quadcopt.py`**: The main training script. It sets up the PPO agent (Actor-Critic network), initializes the environment, and runs the training loop with a GUI progress bar.
* **`quad_env.py`**: A custom OpenAI Gym-compatible environment. It handles the drone's physics (using PyBullet), reward calculation, and state observation.
* **`quadrotor.urdf`** / **`quadrotor_base.obj`**: 3D model files used by PyBullet to render the drone (ensure paths in `quad_env.py` match your local setup).
* **`requirements.txt`**: List of Python dependencies required to run the project.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Matteo7100/Drone-Falco-Reinforcement-Learning.git](https://github.com/Matteo7100/Drone-Falco-Reinforcement-Learning.git)
    cd Drone-Falco-Reinforcement-Learning
    ```

2.  **Install dependencies:**
    Install the required libraries using the provided requirements file:
    ```bash
    pip install -r requirements.txt
    ```

    *Note for Linux Users: If you encounter an error regarding `tkinter`, you may need to install it via your package manager (e.g., `sudo apt-get install python3-tk`).*

### Configuration (Important)

Before running, open **`quad_env.py`** and check line 24:
```python
self.drone = p.loadURDF(r"C:\Users\matte\Desktop\FALCO\Drone_Falco\quadrotor.urdf")

#### 🤝 Contributing
Contributions are welcome! If you find bugs or want to improve the reward function or drone dynamics:

- Fork the Project
- Create your Feature Branch (git checkout -b feature/AmazingFeature)
- Commit your Changes (git commit -m 'Add some AmazingFeature')
- Push to the Branch (git push origin feature/AmazingFeature)
- Open a Pull Request

##### 👤 Author
Matteo7100 (GitHub Profile)
