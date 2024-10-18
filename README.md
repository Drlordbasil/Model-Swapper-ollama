# Model Swapper with Adaptive Swapping and Reflection Loops

This project implements a model swapper that dynamically selects and switches between different machine learning models using reinforcement learning techniques. The core features are implemented in Python with proper module imports.

## Features

- Adaptive model selection using Deep Q-Network (DQN)
- Supports multiple models for text generation and embedding via Ollama
- Incorporates tool executions for code analysis and optimization
- Reinforcement learning for optimizing model selection based on performance metrics

## Requirements

- Python 3.12
- Ollama local server
- Required Python packages listed in `requirements.txt`

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Drlordbasil/Model-Swapper-ollama.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd Model-Swapper-ollama
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python main.py
```

## License
This project is licensed under the MIT License.

