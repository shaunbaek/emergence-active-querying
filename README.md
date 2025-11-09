# Emergence Spatial

## Setup Instructions

### 1. Create a Virtual Environment

First, create a Python virtual environment to isolate project dependencies:

```bash
python3 -m venv .venv
```

> **Note:** You can rename `.venv` to any name you prefer (e.g., `venv`, `env`, etc.)

### 2. Activate the Virtual Environment

Activate the virtual environment using:

```bash
source .venv/bin/activate
```

> **Note:** On Windows, use `.venv\Scripts\activate` instead

### 3. Install Dependencies

Once the virtual environment is activated, upgrade pip and install the required packages:

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install project dependencies
pip install ai2thor matplotlib google-generativeai
```

### 4. Running the Project

After completing the setup, you can run the project scripts:

```bash
python loop.py
# or
python policy_analysis.py
# or
python policy.py
```

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```
