# OC-Multi-Param-AM
This repo contains the minimal code to reproduce the tests for the following paper:
"Optimal multi-parameter control of trapped active matter" L. K. Davis (2026) arXiv: https://arxiv.org/abs/2603.16778

Dataset containing protocols found in "Learning protocols for the fast and efficient control of active matter" Casert et al. Nat. Comm. (2024) are located at [Springer Extra Supplementary Material](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-52878-2/MediaObjects/41467_2024_52878_MOESM4_ESM.zip).

Once you have the correct packages installed. You may run the following:

To produce data/plots for Fig. 1:
python3 adam_optimizer_schmiedl_test_JAX_v2.py
(For unregularized control the regularization parameter is to 0.0).

To produce data/plots for Fig. 2:
python3 adam_optimizer_closed-loop-control_schuttler_test_v4.py

To produce data/plots for Fig S1:
No state-to-state transformation-

python3 adam_optimizer_active-NOs2s-casert_test_JAX_v3.py

Stata-to-state transformation-

python3 adam_optimizer_active-s2s-casert_test_JAX_v2.py
---

## Setup Instructions

To run the optimization codes for the single-parameter testing scenarios (Schmiedl-Seifert, Schüttler, Casert), you will need to set up a Python environment with the required dependencies.

### 1. Clone the Repository
First, clone this repository to your local machine and navigate into the project directory:
```bash
git clone [https://github.com/yourusername/OC-Multi-Param-AM.git](https://github.com/yourusername/OC-Multi-Param-AM.git)
cd OC-Multi-Param-AM
2. Create a Virtual Environment

It is highly recommended to use a virtual environment to keep your dependencies isolated. Run the following command in your terminal to create a virtual environment named venv:

Bash
python -m venv venv
(Note: You may need to use python3 depending on your system configuration.)

3. Activate the Virtual Environment

Activate the environment based on your operating system:

macOS and Linux:

Bash
source venv/bin/activate
Windows:

Bash
venv\Scripts\activate
Once activated, your terminal prompt will change to show (venv) at the beginning of the line.

4. Install Dependencies

With the virtual environment active, ensure pip is up to date, and then install the required packages:

Bash
pip install --upgrade pip
pip install jax matplotlib numpy scipy
Note on JAX: The command above installs the CPU-only version of JAX, which is generally sufficient for basic testing. If you have a compatible GPU and want to run the optimizations with hardware acceleration, please refer to the official JAX installation guide for specific CUDA/ROCm setup instructions.
