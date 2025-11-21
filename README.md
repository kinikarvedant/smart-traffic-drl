# Smart Adaptive Traffic Flow Optimization using DRL

**Short:** trains a PPO agent (Traffic-Pressure reward) to control a 4-way intersection in SUMO.

---

## ▶️ Quick status
- SUMO 1.25.0 tested
- Trainer: Stable-Baselines3 (PPO) + PyTorch
- Environment: `envs/sumo_env.py` (Gymnasium-compatible)
- Evaluator: `agents/eval.py` (auto-loads latest checkpoint)
- TensorBoard logs: `results/ppo_logs/`

---

## Requirements
- Windows 10/11
- Python 3.10+ (venv recommended)
- SUMO 1.25+ (add `.../Eclipse/Sumo/bin` to PATH)
- NVIDIA GPU optional (RTX 4xxx series tested) — install CUDA-enabled PyTorch if you want GPU

Install Python packages:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# (if using GPU) pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
