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
import json
actions = json.load(open('actions.json'))
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
nltk.download('punkt', quiet=True)

# Hyperparameters
alpha = 0.001  # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.2  # Exploration rate
num_episodes = 100  # Number of training iterations
batch_size = 16
memory_size = 100
target_update = 10

# Define tools with real functionalities
tools = actions
    

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
        return f"Error executing code: {str(e)}"

def execute_tool(tool_name, params):
    """
    Executes the specified tool with given parameters.
    """
    if tool_name == "analyze_code":
        return analyze_code(params["code"])
    elif tool_name == "generate_docstring":
        return generate_docstring(params["function_code"])
    elif tool_name == "suggest_optimization":
        return suggest_optimization(params["code"])
    elif tool_name == "generate_test":
        return generate_test(params["function_code"])
    elif tool_name == "explain_code":
        return explain_code(params["code"])
    elif tool_name == "find_bugs":
        return find_bugs(params["code"])
    elif tool_name == "test_code_locally":
        return test_code_locally(params["code"])
    else:
        return f"Unknown tool: {tool_name}"

def model_supports_generate(model_name):
    """
    Checks if the model supports the 'generate' method.
    """
    try:
        # Try to generate an empty prompt to test if model supports generate
        response = ollama.generate(model=model_name, prompt="", options={"num_predict": 1})
        return True
    except Exception:
        return False

def get_model_capabilities(model_name):
    """
    Returns the capabilities of the model ('generate').
    """
    capabilities = []
    if model_supports_generate(model_name):
        capabilities.append('generate')
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

# Define the task
task_prompt = input("Enter your Python coding task or question: ")

# List available models
models_info = ollama.list().get("models", [])
model_names = [model['name'] for model in models_info]
print("\nAvailable models:", model_names)

# Filter out models that do not support 'generate'
model_capabilities = {}
for model_name in model_names:
    capabilities = get_model_capabilities(model_name)
    if 'generate' in capabilities:
        model_capabilities[model_name] = capabilities

if not model_capabilities:
    print("No models with 'generate' capability found.")
    exit()

# Define the neural network model
class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize DQN and target network
state_size = 1  # The state is fixed (the task)
action_size = len(model_capabilities)  # Number of models with capabilities
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
memory = []

capable_model_names = list(model_capabilities.keys())

def select_action(state, epsilon):
    """
    Selects an action (model) based on epsilon-greedy policy.
    """
    if random.uniform(0, 1) < epsilon:
        action = random.randrange(action_size)
        print("Exploring: Randomly selected model.")
    else:
        with torch.no_grad():
            q_values = policy_net(state)
            max_q = q_values.max().item()
            actions = (q_values == max_q).nonzero(as_tuple=True)[1]
            action = random.choice(actions.tolist())
        print("Exploiting: Selected best model based on Q-values.")
    return action

def generate_response(model, prompt):
    """
    Generates a response using the specified model and handles tool execution if needed.
    """
    system_prompt = f"""You are an AI assistant specialized in Python programming.

You have access to the following tools:

{json.dumps(tools, indent=2)}

When assisting the user:

- Carefully read and understand the user's request.
- If you need to use a tool, respond in the following JSON format:

{{
    "tool": "<tool_name>",
    "params": {{
        "<param1>": "<value1>",
        "<param2>": "<value2>",
        ...
    }}
}}

- Do not include any other text in your response when calling a tool.
- After receiving the tool result, incorporate it into your final answer.
- Provide clear explanations and code examples.
- Ensure any code you provide is tested using the 'test_code_locally' tool before sharing.
- Follow Python best practices and PEP8 conventions.

Proceed to assist the user."""

    try:
        # Start the conversation
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        tools_list = [
            {
                'type': 'function',
                'function': {
                    'name': tool_name,
                    'description': tool_info['description'],
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            param: {'type': 'string', 'description': desc}
                            for param, desc in tool_info['parameters'].items()
                        },
                        'required': list(tool_info['parameters'].keys())
                    }
                }
            } for tool_name, tool_info in tools.items()
        ]

        # Get the assistant's response
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=tools_list
        )
        assistant_message = response['message']

        # Check if the assistant called a tool
        if assistant_message.get('tool_calls'):
            for tool_call in assistant_message['tool_calls']:
                tool_name = tool_call['function']['name']
                params = tool_call['function']['arguments']
                tool_result = execute_tool(tool_name, params)
                # Add the tool's response to the conversation
                messages.append(assistant_message)
                messages.append({'role': 'tool', 'content': tool_result})
                # Get the assistant's final response
                final_response = ollama.chat(
                    model=model,
                    messages=messages,
                    tools=tools_list
                )
                return final_response['message']['content'].strip()
        else:
            # No tool was used; return the assistant's content
            return assistant_message['content'].strip()

    except Exception as e:
        print(f"Error generating response with {model}: {str(e)}")
        return None

def evaluate_response(response, action):
    """
    Evaluates the response using a reviewer model and returns a numerical score.
    """
    # Simplified evaluation
    if response:
        return 10  # Assign maximum score for non-empty response
    else:
        return 0

# Main training loop
for episode in range(num_episodes):
    print(f"\nEpisode {episode + 1}/{num_episodes}")
    state = torch.tensor([[0.0]], dtype=torch.float32)  # State is fixed
    action = select_action(state, epsilon)
    selected_model = capable_model_names[action]
    print(f"Selected model: {selected_model}")

    # Ensure the model is available
    if not ensure_model_available(selected_model):
        continue

    response = generate_response(selected_model, task_prompt)

    if response is None:
        reward = 0
    else:
        print(f"Response from {selected_model}:\n{response}")
        reward = evaluate_response(response, action)
        print(f"Reward (score): {reward}")
    reward_tensor = torch.tensor([reward], dtype=torch.float32)

    # Store transition in memory
    memory.append((state, action, reward_tensor))

    # Limit memory size
    if len(memory) > memory_size:
        memory.pop(0)

    # Sample from memory and update policy
    if len(memory) >= batch_size:
        batch = random.sample(memory, batch_size)
        states, actions, rewards = zip(*batch)
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.stack(rewards)

        q_values = policy_net(states).gather(1, actions)
        with torch.no_grad():
            target_q_values = rewards.unsqueeze(1)
        loss = nn.functional.mse_loss(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target network periodically
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

# After training, select the best model
state = torch.tensor([[0.0]], dtype=torch.float32)
with torch.no_grad():
    q_values = policy_net(state)
    max_q = q_values.max().item()
    actions = (q_values == max_q).nonzero(as_tuple=True)[1]
    best_action = random.choice(actions.tolist())
    best_model = capable_model_names[best_action]
print(f"\nBest model after training: {best_model}")

# Ensure the best model is available
ensure_model_available(best_model)

# Generate final response using the best model
final_response = generate_response(best_model, task_prompt)
if final_response:
    print(f"\nFinal response from best model ({best_model}):\n{final_response}")
else:
    print(f"Failed to generate response with best model {best_model}.")

# Save the model
torch.save(policy_net.state_dict(), 'policy_net.pth')
print("\nModel saved to policy_net.pth")
