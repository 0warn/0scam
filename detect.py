import sys
import os
import joblib
import numpy as np
import pandas as pd
from features.feature_extractor import extract_features
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
import time # For simulating LLM delay

# Rich library imports for attractive CLI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.rule import Rule
from rich.status import Status # Import for animated status messages
from rich.align import Align # Import for alignment
from rich.layout import Layout # For more structured UI
from rich.live import Live # For dynamic updating chat history
from rich.columns import Columns # For arranging user/AI messages

# Prompt_toolkit imports for advanced input
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.input import create_input # For explicit input/output
from prompt_toolkit.output import create_output




# --- LLM Integration Placeholder ---
# NOTE: This is a placeholder. For real functionality, you would integrate with an actual LLM API
# (e.g., Google's Gemini API, OpenAI API). This requires API keys and handling API responses.
def get_llm_response(user_message, console):
    """Simulates an LLM response or acts as a placeholder for actual LLM API integration."""
    with console.status("[italic dim]AI is thinking...[/italic dim]", spinner="point") as status: # Use status for temporary message
        time.sleep(2) # Simulate processing time

    # Simple keyword-based responses for demonstration
    user_message_lower = user_message.lower()
    if "hello" in user_message_lower or "hi" in user_message_lower:
        return "Hello there! How can I help you today? Do you have a URL you'd like me to check?"
    elif "how are you" in user_message_lower:
        return "I'm an AI, so I don't have feelings, but I'm ready to assist you!"
    elif "thank you" in user_message_lower or "thanks" in user_message_lower:
        return "You're welcome! Let me know if you need anything else."
    elif "what can you do" in user_message_lower or "help" in user_message_lower:
        return "I can detect phishing URLs and engage in a basic conversation. Try giving me a URL!"
    else:
        return "I'm not sure how to respond to that, but I can check URLs for phishing. Just paste one here!"

# --- URL Availability Check ---
def check_url_availability(url):
    """
    Checks the availability of a URL and returns its HTTP status code and reachability.
    Handles various network errors.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        # Use a reasonable timeout to prevent hanging
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        return response.status_code, "Reachable"
    except ConnectionError:
        return None, "Connection Error (Host Down/Unreachable)"
    except Timeout:
        return None, "Timeout (Took too long to respond)"
    except RequestException as e:
        return None, f"Request Error: {e}"
    except Exception as e:
        return None, f"Unknown Error: {e}"

# --- Phishing Detection Logic ---
def detect_phishing_url(url, model_path_override=None, scaler_path='model/scaler.joblib'):
    """
    Detects if a given URL is legitimate, suspicious, or phishing using a trained ML model and scaler.
    Renamed to avoid conflict with main detect.py purpose.
    """
    # Determine which model to load
    model_to_load = model_path_override if model_path_override else 'model/logistic_regression_model.joblib'

    try:
        model = joblib.load(model_to_load)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        return "Error", "HIGH", 0.0, f"Model '{model_to_load}' or Scaler '{scaler_path}' not found. Please train the model first by running train.sh.", None
    except Exception as e:
        return "Error", "HIGH", 0.0, f"Error loading model/scaler: {e}", None

    # Extract features from the input URL
    features_dict = extract_features(url)

    # Ensure the feature vector has the same order and names as during training
    dummy_features = extract_features("http://example.com")
    feature_names = list(dummy_features.keys())
    
    # Convert to DataFrame before scaling to retain feature names
    feature_df = pd.DataFrame([features_dict.get(name, 0) for name in feature_names]).T
    feature_df.columns = feature_names

    # Scale the feature vector using the loaded scaler
    feature_vector_scaled = scaler.transform(feature_df) # Pass DataFrame to scaler

    # Predict probability and class
    probabilities = model.predict_proba(feature_vector_scaled)[0]
    phishing_probability = probabilities[1] # Probability of being phishing

    prediction = model.predict(feature_vector_scaled)[0]

    classification_result = ""
    risk_level = ""

    PHISHING_THRESHOLD = 0.75
    SUSPICIOUS_THRESHOLD = 0.3

    if prediction == 1: # Model predicted phishing
        classification_result = "Phishing"
        if phishing_probability >= 0.95:
            risk_level = "HIGH"
        elif phishing_probability >= PHISHING_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "MEDIUM"
    else: # Model predicted legitimate
        if phishing_probability >= SUSPICIOUS_THRESHOLD:
            classification_result = "Suspicious"
            risk_level = "MEDIUM"
        else:
            classification_result = "Legitimate"
            risk_level = "LOW"

    confidence_score = phishing_probability * 100

    return classification_result, risk_level, confidence_score, None, features_dict # Return features_dict for heuristic

def is_valid_url(url):
    """Basic validation to check if a string is a well-formed URL."""
    try:
        result = urlparse(url)
        # Check for both scheme and netloc (domain/IP) to be valid
        return all([result.scheme, result.netloc]) and ("." in result.netloc or re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", result.netloc))
    except ValueError:
        return False

# Function to render chat history for rich.Live
def make_chat_history_renderable(history):
    renderables = []
    # Only show last few messages to avoid overwhelming the screen
    for entry in history[-8:]: # Show last 8 messages for a good balance
        if entry["role"] == "user":
            renderables.append(
                Panel(Text(entry["content"], style="cyan"), title="[bold cyan]You[/bold cyan]", border_style="cyan", expand=True, title_align="left")
            )
        else: # AI response
            if isinstance(entry["content"], Panel): # If content is a Panel (URL analysis result)
                renderables.append(entry["content"]) # Print the Panel directly
            else: # If content is a string (LLM response)
                renderables.append(Panel(Text(entry["content"], style="magenta"), title="[bold magenta]AI[/bold magenta]", border_style="magenta", expand=True, title_align="left"))
        # Add some space between messages
        renderables.append(Text("\n"))
    return Columns(renderables, padding=1) # Use Columns to arrange messages

# Helper function to print chat history and welcome message
def _print_chat_history(console, welcome_panel_content_text, chat_history):
    console.clear() # Clear screen at the start of each turn

    # Print welcome message
    console.print(welcome_panel_content_text)

    # Print chat history
    for entry in chat_history[-8:]: # Show last 8 messages
        if entry["role"] == "user":
            console.print(
                Panel(Text(entry["content"], style="cyan"), title="[bold cyan]You[/bold cyan]", border_style="cyan", expand=True, title_align="left")
            )
        else: # AI response
            if isinstance(entry["content"], Panel):
                console.print(entry["content"])
            else:
                console.print(Panel(Text(entry["content"], style="magenta"), title="[bold magenta]AI[/bold magenta]", border_style="magenta", expand=True, title_align="left"))
        console.print(Text("\n")) # Add some space

    console.print(Rule(characters="═", style="blue")) # Separator before input


# --- Main Conversational Loop with Rich UI ---



def main():
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        try:
            joblib.load('model/random_forest_model.joblib')
            model_path = 'model/random_forest_model.joblib'
        except FileNotFoundError:
            try:
                joblib.load('model/logistic_regression_model.joblib')
                model_path = 'model/logistic_regression_model.joblib'
            except FileNotFoundError:
                console.print(Panel(
                    Text("Error: No trained model found. Please run [bold yellow]train.sh[/bold yellow] first.", justify="center"),
                    title="[bold red]Model Error[/bold red]",
                    border_style="red"
                ))
                sys.exit(1)

    # console.clear() # Clear terminal on start
    # Welcome Panel Content
    welcome_panel_content_text = Text.from_markup(
        f"✨ [bold green]Phishing URL Detector & Chat Assistant[/bold green] ✨\n"
        f"Using model: [bold cyan]{model_path}[/bold cyan]\n"
        f"I'm ready to chat! Ask me anything or paste a URL for analysis.\n"
        f"Type '[bold yellow]exit[/bold yellow]' to quit."
    )

    # Chat history
    chat_history = []

    # Initialize prompt_toolkit session
    # Explicitly create input/output for prompt_toolkit to try and improve compatibility


    pt_style = Style.from_dict({
        'prompt': 'bold yellow',
        'rprompt': 'ansicyan',
        'text': '#ffffff',
        'bottom-toolbar': 'reverse ansigreen',
        'bottom-toolbar.text': '#ffffff',
    })
    pt_input = create_input()
    pt_output = create_output()

    session = PromptSession(
        style=pt_style,
        input=pt_input, # Pass explicit input
        output=pt_output, # Pass explicit output
    )

    console = Console()

    while True:
        _print_chat_history(console, welcome_panel_content_text, chat_history)

        try:
            user_input = session.prompt(HTML('<ansiyellow><b>🚀 You:</b></ansiyellow> ')).strip()

            if user_input.lower() == 'exit':
                console.print(Text("\nExiting. Goodbye! 👋", style="bold green"))
                break
            
            if not user_input:
                continue

            chat_history.append({"role": "user", "content": user_input})

            # --- Intent Recognition ---
            if " " not in user_input and (user_input.startswith("http://") or user_input.startswith("https://") or user_input.startswith("ftp://")):
                # User provided a URL, invoke detection
                with console.status("[italic dim]AI is processing the URL...[/italic dim]", spinner="point") as status:
                    status_icon_val = "❓" # Initialize
                    status_color_val = "grey"
                    reachability_icon_val = "❓"
                    reachability_color_val = "grey"
                    status_code_val, reachability_val = check_url_availability(user_input)

                    status_icon_val = "✅" if status_code_val and 200 <= status_code_val < 400 else "⚠️" if status_code_val and 400 <= status_code_val < 500 else "❌"
                    status_color_val = "green" if status_code_val and 200 <= status_code_val < 400 else "yellow" if status_code_val and 400 <= status_code_val < 500 else "red"
                    reachability_icon_val = "✅" if reachability_val == "Reachable" else "❌"
                    reachability_color_val = "green" if reachability_val == "Reachable" else "red"

                with console.status("[italic magenta]🧠 Analyzing URL for phishing characteristics...[/italic magenta]", spinner="line") as status:
                    detection_results = detect_phishing_url(user_input, model_path)
                    
                    if len(detection_results) == 5:
                        classification, risk, confidence, error_message, features_dict_for_heuristic = detection_results
                    else:
                        classification, risk, confidence, error_message = detection_results
                        features_dict_for_heuristic = extract_features(user_input)

                if error_message:
                    ai_response = f"Error during URL detection: {error_message}"
                else:
                    if (classification == "Legitimate" or classification == "Suspicious") and reachability_val != "Reachable":
                        if features_dict_for_heuristic.get('is_suspicious_tld', 0) == 1 or features_dict_for_heuristic.get('has_uncommon_tld', 0) == 1:
                            if features_dict_for_heuristic.get('is_suspicious_tld', 0) == 1:
                                classification = "Phishing"
                                risk = "HIGH"
                                confidence = max(confidence, 70.0)
                                heuristic_message = "📢 Heuristic applied: Unreachable URL with [bold red]suspicious TLD[/bold red] detected. Elevated to Phishing."
                            else:
                                classification = "Suspicious"
                                risk = "MEDIUM"
                                confidence = max(confidence, 50.0)
                                heuristic_message = "📢 Heuristic applied: Unreachable URL with [bold yellow]uncommon TLD[/bold yellow] detected. Elevated to Suspicious."
                            chat_history.append({"role": "ai", "content": Text(heuristic_message, style="bold yellow")})
                    
                    result_table = Table(
                        style="cyan", box=box.ROUNDED, show_header=False, width=None
                    )
                    result_table.add_column("Property", justify="right", style="bold blue")
                    result_table.add_column("Value", justify="left")

                    class_color = "red" if classification == "Phishing" else "yellow" if classification == "Suspicious" else "green"
                    class_icon = "🎣" if classification == "Phishing" else "🧐" if classification == "Suspicious" else "✅"
                    risk_color = "red" if risk == "HIGH" else "yellow" if risk == "MEDIUM" else "green"
                    risk_icon = "🔥" if risk == "HIGH" else "⚠️" if risk == "MEDIUM" else "🛡️"

                    result_table.add_row(Text("Classification:", style="bold"), Text(f"{class_icon} {classification}", style=class_color))
                    result_table.add_row(Text("Risk Level:", style="bold"), Text(f"{risk_icon} {risk}", style=risk_color))
                    result_table.add_row(Text("Confidence:", style="bold"), Text(f"{confidence:.2f}%", style="bold magenta"))
                    result_table.add_row(Text("HTTP Status:", style="bold"), Text(f"{status_icon_val} {status_code_val if status_code_val else 'N/A'}", style=status_color_val))
                    result_table.add_row(Text("Reachability:", style="bold"), Text(f"{reachability_icon_val} {reachability_val}", style=reachability_color_val))
                    
                    ai_response = Panel(
                        result_table,
                        title=Text(f"Analysis for: [bold white]{user_input}[/bold white]", style="bold blue", justify="left"),
                        border_style="green", box=box.HEAVY, expand=True, padding=(1, 2)
                    )
                chat_history.append({"role": "ai", "content": ai_response})

            else:
                ai_response_text = get_llm_response(user_input, console)
                chat_history.append({"role": "ai", "content": ai_response_text})

        except EOFError:
            console.print(Text("\nExiting due to EOF (Ctrl+D). Goodbye! 👋", style="bold green"))
            break
        except KeyboardInterrupt:
            console.print(Text("\nExiting due to user interruption (Ctrl+C). Goodbye! 👋", style="bold green"))
            break
        except Exception as e:
            error_panel_content = Text(f"An unexpected error occurred: {e}", justify="center", style="bold red")
            console.print(Panel(
                error_panel_content,
                title="[bold red]Runtime Error[/bold red]",
                border_style="red"
            ))
            chat_history.append({"role": "ai", "content": error_panel_content})
            break # Exit cleanly on unexpected errors

    # console.print(Text("\nExiting. Goodbye! 👋", style="bold green")) # Moved inside exit condition



if __name__ == '__main__':
    main()
