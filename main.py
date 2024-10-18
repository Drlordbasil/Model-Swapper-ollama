import ollama
import random
import re
import json
import ast
import subprocess
import sys
import tempfile
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
from sklearn.neighbors import NearestNeighbors
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hyperparameters
ALPHA = 0.001        # Learning rate for optimizer
GAMMA = 0.9          # Discount factor (not used currently)
EPSILON = 0.2        # Initial exploration rate
MIN_EPSILON = 0.01   # Minimum exploration rate
DECAY_RATE = 0.995   # Decay rate for epsilon
NUM_EPISODES = 5   # Number of training iterations
BATCH_SIZE = 16
MEMORY_SIZE = 100
TARGET_UPDATE = 10

# Define tools with real functionalities
tools = {
    "analyze_code": {
        "description": "Analyze Python code for syntax errors, PEP8 compliance, and provide basic statistics.",
        "parameters": {
            "code": "The Python code to analyze."
        }
    },
    "generate_docstring": {
        "description": "Generate a detailed docstring for a Python function, including parameters and return types.",
        "parameters": {
            "function_code": "The Python function code."
        }
    },
    "suggest_optimization": {
        "description": "Suggest performance optimizations and best practices for a given Python code snippet.",
        "parameters": {
            "code": "The Python code to optimize."
        }
    },
    "generate_test": {
        "description": "Generate unit tests for a given Python function using unittest or pytest frameworks.",
        "parameters": {
            "function_code": "The Python function code to test."
        }
    },
    "explain_code": {
        "description": "Provide a line-by-line explanation of what the given Python code does.",
        "parameters": {
            "code": "The Python code to explain."
        }
    },
    "find_bugs": {
        "description": "Analyze Python code to find logical errors or potential bugs.",
        "parameters": {
            "code": "The Python code to analyze for bugs."
        }
    },
    "test_code_locally": {
        "description": "Execute Python code locally and capture any exceptions or output.",
        "parameters": {
            "code": "The Python code to execute."
        }
    }
}

def analyze_code(code):
    """
    Analyzes Python code for syntax errors and PEP8 compliance.
    Returns syntax check result and basic statistics.
    """
    try:
        ast.parse(code)
        lines = code.strip().split('\n')
        chars = len(code)
        return f"Code is syntactically correct. Lines: {len(lines)}, Characters: {chars}"
    except SyntaxError as e:
        return f"Syntax error: {str(e)}"

def generate_docstring(function_code):
    """
    Generates a detailed docstring for a Python function.
    """
    try:
        # Extract function signature
        tree = ast.parse(function_code)
        func_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        params = [arg.arg for arg in func_node.args.args]
        docstring = f'"""{func_node.name} function.\n\n'
        if params:
            docstring += "Parameters:\n"
            for param in params:
                docstring += f"    {param} : type\n        Description of {param}.\n"
        docstring += "\nReturns:\n    type\n        Description of return value.\n\"\"\""
        return docstring
    except Exception as e:
        return f"Error generating docstring: {str(e)}"

def suggest_optimization(code):
    """
    Suggests performance optimizations and best practices for Python code.
    """
    suggestions = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call) and getattr(node.iter.func, 'id', '') == 'range':
                    if isinstance(node.iter.args[0], ast.Call):
                        suggestions.append("Consider using enumerate() for better readability when iterating with indices.")
                suggestions.append("Use list comprehensions where appropriate for more concise code.")
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 50:
                    suggestions.append(f"Function '{node.name}' is too long; consider splitting it into smaller functions.")
            if isinstance(node, ast.ImportFrom) and node.module == '__future__':
                suggestions.append("Ensure compatibility with future Python versions.")
        if not suggestions:
            return "Code follows best practices."
        else:
            return "\n".join(set(suggestions))
    except Exception as e:
        return f"Error analyzing code for optimizations: {str(e)}"

def generate_test(function_code):
    """
    Generates unit tests for a given Python function.
    """
    try:
        tree = ast.parse(function_code)
        func_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        test_code = f"""import unittest

{function_code}

class Test{func_node.name.capitalize()}(unittest.TestCase):
    def test_{func_node.name}_example(self):
        # Example test case
        result = {func_node.name}()  # Add necessary arguments
        expected = None  # Define the expected result
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
"""
        return test_code
    except Exception as e:
        return f"Error generating test: {str(e)}"

def explain_code(code):
    """
    Provides a line-by-line explanation of Python code.
    """
    explanations = []
    lines = code.strip().split('\n')
    for i, line in enumerate(lines, 1):
        explanations.append(f"Line {i}: {line.strip()} - Explanation of what this line does.")
    return "\n".join(explanations)

def find_bugs(code):
    """
    Analyzes code to find logical errors or potential bugs.
    """
    bugs = []
    try:
        tree = ast.parse(code)
        # Simple check for division by zero
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.right, ast.Num) and node.right.n == 0:
                    bugs.append("Division by zero detected.")
        if not bugs:
            return "No obvious bugs found."
        else:
            return "\n".join(set(bugs))
    except Exception as e:
        return f"Error analyzing code for bugs: {str(e)}"

def test_code_locally(code):
    """
    Executes Python code locally and captures any exceptions or output.

    Args:
        code (str): The Python code to execute.

    Returns:
        str: Execution output or error message.

    Raises:
        RuntimeError: If there is an error during code execution.
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        # Execute the code and capture output
        result = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True)

        os.remove(temp_file_path)  # Clean up the temporary file

        if result.returncode != 0:
            return f"Execution error:\n{result.stderr}"
        else:
            return f"Code executed successfully. Output:\n{result.stdout}"
    except Exception as e:
        raise RuntimeError(f"Error executing code: {str(e)}")

def execute_tool(tool_name, params):
    """
    Executes the specified tool with given parameters.

    Args:
        tool_name (str): The name of the tool to execute.
        params (dict): The parameters required for the tool.

    Returns:
        str: The output from the tool execution.

    Raises:
        ValueError: If the tool name is unknown.
    """
    tools_mapping = {
        "analyze_code": analyze_code,
        "generate_docstring": generate_docstring,
        "suggest_optimization": suggest_optimization,
        "generate_test": generate_test,
        "explain_code": explain_code,
        "find_bugs": find_bugs,
        "test_code_locally": test_code_locally,
    }
    if tool_name in tools_mapping:
        return tools_mapping[tool_name](**params)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

def does_model_support_generate(model_name):
    """
    Checks if the model supports the 'generate' method.

    Args:
        model_name (str): The name of the model to check.

    Returns:
        bool: True if the model supports 'generate', False otherwise.
    """
    try:
        # Attempt to generate an empty prompt to test if the model supports 'generate'
        ollama.generate(model=model_name, prompt="", options={"num_predict": 1})
        return True
    except Exception:
        return False

def does_model_support_embed(model_name):
    """
    Checks if the model supports the 'embed' method.

    Args:
        model_name (str): The name of the model to check.

    Returns:
        bool: True if the model supports 'embed', False otherwise.
    """
    try:
        # Attempt to embed an empty string to test if the model supports 'embed'
        ollama.embed(model=model_name, input="")
        return True
    except Exception:
        return False

def get_model_capabilities(model_name):
    """
    Returns the capabilities of the model ('generate', 'embed', or both).
    """
    capabilities = []
    if does_model_support_generate(model_name):
        capabilities.append('generate')
    if does_model_support_embed(model_name):
        capabilities.append('embed')
    return capabilities

def ensure_model_available(model_name):
    """
    Ensures that the model is available by pulling it if necessary.
    """
    available_models = [model['name'] for model in ollama.list().get("models", [])]
    if model_name not in available_models:
        print(f"Model {model_name} not found locally. Attempting to pull the model...")
        try:
            ollama.pull(model_name)
            print(f"Successfully pulled model {model_name}.")
        except Exception as e:
            print(f"Failed to pull model {model_name}: {str(e)}")
            return False
    return True

def generate_embeddings(text_chunks, embedding_model):
    """
    Generates embeddings for the given text chunks using the specified embedding model.

    Args:
        text_chunks (list of str): The text chunks to generate embeddings for.
        embedding_model (str): The name of the embedding model to use.

    Returns:
        list: A list of embeddings.

    Raises:
        ValueError: If embeddings could not be generated.
    """
    try:
        response = ollama.embed(model=embedding_model, input=text_chunks)
        embeddings = response.get("embeddings")
        if not embeddings:
            raise ValueError("No embeddings returned by the model.")
        return embeddings
    except Exception as e:
        raise ValueError(f"Error generating embeddings with {embedding_model}: {str(e)}")

def chunk_text(text, max_length=512):
    """
    Splits text into chunks suitable for embedding.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def load_config():
    """
    Loads hyperparameters and configuration settings.

    Returns:
        dict: A dictionary containing configuration parameters.
    """
    config = {
        'alpha': 0.001,       # Learning rate for optimizer
        'gamma': 0.9,         # Discount factor (not used currently)
        'epsilon': 0.2,       # Exploration rate
        'min_epsilon': 0.01,  # Minimum exploration rate
        'decay_rate': 0.995,  # Decay rate for epsilon
        'num_episodes': 100,  # Number of training iterations
        'batch_size': 16,
        'memory_size': 100,
        'target_update': 10
    }
    return config

def main():
    """
    Main function to execute the RL agent for model selection.
    """
    # Load configuration
    config = load_config()
    alpha = config['alpha']
    epsilon = config['epsilon']
    num_episodes = config['num_episodes']
    batch_size = config['batch_size']
    memory_size = config['memory_size']
    target_update = config['target_update']

    # Define the task
    task_prompt = input("Enter your Python coding task or question: ")

    # Initialize models and capabilities
    models_info = ollama.list().get("models", [])
    model_names = [model['name'] for model in models_info]
    model_capabilities = get_models_capabilities(model_names)

    # Filter models that support 'generate'
    capable_model_names = [name for name, caps in model_capabilities.items() if 'generate' in caps]

    if not capable_model_names:
        print("No models with 'generate' capability found.")
        return

    # Initialize RL components
    state_size = 1  # The state is fixed (the task)
    action_size = len(capable_model_names)
    policy_net, target_net, optimizer = initialize_rl_components(state_size, action_size, alpha)
    memory = []

    # Start training loop
    training_loop(task_prompt, capable_model_names, policy_net, target_net,
                  optimizer, memory, config, model_capabilities)

def training_loop(task_prompt, capable_model_names, policy_net, target_net,
                  optimizer, memory, config, model_capabilities):
    """
    Runs the main training loop.

    Args:
        task_prompt (str): The user's task or question.
        capable_model_names (list): List of models capable of 'generate'.
        policy_net (nn.Module): The policy network.
        target_net (nn.Module): The target network.
        optimizer (torch.optim.Optimizer): The optimizer.
        memory (list): Replay memory.
        config (dict): Configuration parameters.
        model_capabilities (dict): Capabilities of the models.
    """
    epsilon = config['epsilon']
    num_episodes = config['num_episodes']
    batch_size = config['batch_size']
    memory_size = config['memory_size']
    target_update = config['target_update']
    min_epsilon = config['min_epsilon']
    decay_rate = config['decay_rate']

    for episode in range(num_episodes):
        logging.info(f"Episode {episode + 1}/{num_episodes}")
        state = torch.tensor([[0.0]], dtype=torch.float32)  # State is static
        action = select_action(state, policy_net, epsilon)
        selected_model = capable_model_names[action]
        logging.info(f"Selected model: {selected_model}")

        # Ensure the model is available
        if not ensure_model_available(selected_model):
            continue

        response = generate_response(selected_model, task_prompt)

        if response is None:
            reward = 0
        else:
            logging.info(f"Response from {selected_model}:\n{response}")
            reward = evaluate_response(response, action, capable_model_names, task_prompt)
            logging.info(f"Reward (score): {reward}")

        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        # Store transition in memory
        memory.append((state, action, reward_tensor))

        # Limit memory size
        if len(memory) > memory_size:
            memory.pop(0)

        # Perform optimization step
        if len(memory) >= batch_size:
            optimize_model(memory, policy_net, optimizer, batch_size)

        # Update target network periodically
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Adjust epsilon
        epsilon = adjust_epsilon(epsilon, min_epsilon, decay_rate)

    # Select and use the best model after training
    use_best_model(task_prompt, capable_model_names, policy_net)

def initialize_rl_components(state_size, action_size, alpha):
    """
    Initializes the policy network, target network, and optimizer.

    Args:
        state_size (int): Size of the state.
        action_size (int): Size of the action space.
        alpha (float): Learning rate.

    Returns:
        tuple: Policy network, target network, and optimizer.
    """
    policy_net = DeepQNetwork(state_size, action_size)
    target_net = DeepQNetwork(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    return policy_net, target_net, optimizer

def optimize_model(memory, policy_net, optimizer, batch_size):
    """
    Performs a single optimization step.
    """
    batch = random.sample(memory, batch_size)
    states, actions, rewards = zip(*batch)
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

    # Compute Q-values and targets
    q_values = policy_net(states).gather(1, actions.unsqueeze(1))
    with torch.no_grad():
        target_q_values = rewards

    # Compute loss and optimize
    loss = nn.functional.mse_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def select_action(state, policy_net, epsilon):
    """
    Selects an action (model index) based on the current policy.

    Args:
        state (torch.Tensor): The current state tensor.
        policy_net (nn.Module): The policy network.
        epsilon (float): The exploration rate.

    Returns:
        int: The index of the selected action.
    """
    if random.uniform(0, 1) < epsilon:
        action = random.randrange(policy_net.fc2.out_features)
        logging.info("Exploring: Randomly selected model.")
    else:
        with torch.no_grad():
            q_values = policy_net(state)
            action = int(q_values.argmax().item())
        logging.info("Exploiting: Selected best model based on Q-values.")
    return action

def generate_response(model, prompt):
    """
    Generates a response using the specified model and handles tool execution if needed.
    """
    # Multi-shot prompting with examples
    system_prompt = f"""You are an AI assistant specialized in Python programming. You have access to the following tools:

{json.dumps(tools, indent=2)}

Guidelines:
- Analyze the user's question carefully.
- If a tool is needed, respond with a JSON object specifying the tool and parameters.
- Provide detailed explanations and code examples.
- Before presenting code to the user, test it locally using the 'test_code_locally' tool.
- Follow Python best practices and PEP8 conventions.
- Structure your response with clear headings and code blocks.

Examples:
1. **User**: How can I optimize this code snippet?
   **Assistant**:
   {{
       "tool": "suggest_optimization",
       "params": {{
           "code": "[User's code]"
       }}
   }}
   [After tool execution, provide optimized code and explanations.]

2. **User**: Can you explain what this function does?
   **Assistant**:
   {{
       "tool": "explain_code",
       "params": {{
           "code": "[User's function]"
       }}
   }}
   [After tool execution, provide line-by-line explanation.]

Now, proceed to assist the user."""

    try:
        # Initial response
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': tool_name,
                        'description': tool_info['description'],
                        'parameters': {
                            'type': 'object',
                            'properties': {param: {'type': 'string', 'description': desc}
                                           for param, desc in tool_info['parameters'].items()},
                            'required': list(tool_info['parameters'].keys()),
                        },
                    },
                } for tool_name, tool_info in tools.items()
            ],
        )
        assistant_message = response['message']
        # Check if the assistant decided to use a tool
        if assistant_message.get('tool_calls'):
            for tool_call in assistant_message['tool_calls']:
                tool_name = tool_call['function']['name']
                params = tool_call['function']['arguments']
                tool_result = execute_tool(tool_name, params)
                # Add the tool's response to the conversation
                response = ollama.chat(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': prompt},
                        assistant_message,
                        {'role': 'tool', 'content': tool_result},
                    ],
                )
                return response['message']['content'].strip()
        else:
            # No tool was used; return the assistant's content
            return assistant_message['content'].strip()

    except Exception as e:
        logging.error(f"Error generating response with {model}: {str(e)}")
        return None

def evaluate_response(response, action, capable_model_names, task_prompt):
    """
    Evaluates the response using reviewer models and returns a numerical score based on multiple criteria.
    """
    reviewer_models = [model for i, model in enumerate(capable_model_names) if i != action]
    if not reviewer_models:
        return 5  # Neutral score if no reviewer models are available
    
    # Use multiple reviewer models for a more robust evaluation
    scores = []
    for reviewer_model in reviewer_models:
        review_prompt = f"""As an expert Python code reviewer, evaluate the following response to the user's task: '{task_prompt}'

Response:
{response}

Criteria for evaluation (score each criterion from 1 to 10):

1. **Task Completion**: Did the response fully address the user's request?
2. **Coherence and Clarity**: Is the response logically structured and easy to understand?
3. **Code Quality**: Are any code examples correct, efficient, and follow best practices?
4. **Length Appropriateness**: Is the response appropriately detailed, neither too brief nor unnecessarily long?
5. **Structure**: Is the response well-organized with proper headings and formatting?

Calculate the average score and provide only the numerical value and a brief justification.

Example:
Score: 8.6
Justification: The response adequately addresses the task with clear explanations and quality code.

Now, provide your evaluation."""
        try:
            review_response = ollama.generate(model=reviewer_model, prompt=review_prompt)
            review_text = review_response['response'].strip()
            # Extract numerical score
            match = re.search(r'Score:\s*([\d\.]+)', review_text)
            if match:
                score = float(match.group(1))
                if 1 <= score <= 10:
                    scores.append(score)
        except Exception as e:
            logging.error(f"Error evaluating response with {reviewer_model}: {str(e)}")
            continue

    if scores:
        # Return the average score from all reviewers
        average_score = sum(scores) / len(scores)
        return average_score
    else:
        return 5  # Neutral score if no valid scores obtained

def adjust_epsilon(epsilon, min_epsilon=0.01, decay_rate=0.995):
    """
    Adjusts the exploration rate epsilon after each episode.

    Args:
        epsilon (float): Current epsilon value.
        min_epsilon (float): Minimum value for epsilon.
        decay_rate (float): Decay rate.

    Returns:
        float: Updated epsilon value.
    """
    epsilon = max(min_epsilon, epsilon * decay_rate)
    return epsilon

def use_best_model(task_prompt, capable_model_names, policy_net):
    """
    Selects the best model after training and uses it to generate a response.

    Args:
        task_prompt (str): The task prompt provided by the user.
        capable_model_names (list): List of capable model names.
        policy_net (nn.Module): The trained policy network.
    """
    state = torch.tensor([[0.0]], dtype=torch.float32)
    with torch.no_grad():
        q_values = policy_net(state)
        best_action = int(q_values.argmax().item())
        best_model = capable_model_names[best_action]
    logging.info(f"Best model after training: {best_model}")

    # Ensure the best model is available
    ensure_model_available(best_model)

    # Generate final response using the best model
    final_response = generate_response(best_model, task_prompt)
    if final_response:
        logging.info(f"Final response from best model ({best_model}):\n{final_response}")
    else:
        logging.error(f"Failed to generate response with best model {best_model}.")

    # Save the trained policy network
    torch.save(policy_net.state_dict(), 'policy_net.pth')
    logging.info("Model saved to policy_net.pth")

def get_models_capabilities(model_names):
    """
    Retrieves the capabilities of each model.

    Args:
        model_names (list): List of model names.

    Returns:
        dict: A dictionary mapping model names to their capabilities.
    """
    model_capabilities = {}
    for model_name in model_names:
        capabilities = get_model_capabilities(model_name)
        if capabilities:
            model_capabilities[model_name] = capabilities
    return model_capabilities

# Define the neural network model
class DeepQNetwork(nn.Module):
    """
    Deep Q-Network model used for approximating the action-value function in Q-learning.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        relu (nn.ReLU): ReLU activation function.
        fc2 (nn.Linear): Second fully connected layer.
    """

    def __init__(self, num_inputs, num_outputs):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Main function
if __name__ == "__main__":
    main()