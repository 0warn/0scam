# Conversational AI Phishing URL Detector

This project has evolved into an **Advanced Conversational AI Assistant** with integrated Phishing URL Detection capabilities. It allows users to interact in a chat-like interface, ask questions, and submit URLs for real-time analysis. The tool aims to be intelligent, responsive, and provide a comprehensive security analysis for URLs, suitable for SOC / blue-team use.

## Project Goal
Transform the Phishing URL detection tool into a conversational AI assistant that can:
-   Engage in natural language conversation (via an integrated LLM).
-   Detect and classify URLs as:
    -   **Legitimate**
    -   **Suspicious**
    -   **Phishing**
-   Provide URL availability and network status.
-   Offer a highly interactive and aesthetically pleasing CLI chat interface using `rich.Live` for dynamic history.

## Functional Requirements
1.  **Accept natural language input:** Via an advanced interactive CLI using `prompt_toolkit`.
2.  **Conversational AI:** Integrates with an LLM (placeholder provided) to respond to natural language queries.
3.  **Intent Recognition:** Automatically determine if user input is a conversational query or a URL for analysis, triggering the appropriate action.
4.  **Extract meaningful lexical and structural features from the URL:** (Expanded feature set).
5.  **Train a machine learning model:** Using a large (simulated or real) phishing URL dataset.
6.  **Use simple, explainable models:** Logistic Regression or Random Forest, with robust training.
7.  **Output:**
    *   AI conversational response, displayed in a chat-like panel.
    *   For URL analysis:
        *   Classification result
        *   Risk level (LOW / MEDIUM / HIGH)
        *   Confidence score (percentage)
        *   URL Availability and HTTP Status Code.
        *   Displayed in a structured `rich.Panel`.
8.  **Interactive Chat UI:** Displays conversation history dynamically with `rich.Live`, now with scrolling history.

## Technical Constraints & Changes
-   **Language:** Python 3
-   **Libraries:** `pandas`, `scikit-learn`, `numpy`, `tldextract`, `requests`, `rich`, `prompt_toolkit`
-   No deep learning for URL classification model.
-   **Relaxation of "No External API Calls" Constraint:** To support conversational AI, integration with an external Large Language Model (LLM) API is now assumed. A placeholder function (`get_llm_response`) is provided, which the user can replace with actual API calls (e.g., Google Gemini API, OpenAI API). This will require API keys.
-   Code must be modular and clean.

## Project Structure
```
.
├── dataset/
│   ├── phishing_dataset.csv         # Large, simulated dataset
│   └── generate_advanced_dataset.py # Script to generate the advanced dummy dataset
├── features/
│   └── feature_extractor.py
├── model/
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   └── scaler.joblib                # Stores the StandardScaler for consistent scaling
├── .venv/
│   └── ... (virtual environment files)
├── detect.py                        # Main conversational AI + URL detector with rich.Live
├── detect.sh                        # Convenience script for launching the assistant
├── train.py                         # Training script for ML model
├── train.sh                         # Convenience script for training
└── README.md
```

## Security & Quality Requirements
-   Validate and sanitize user input (basic validation implemented).
-   Handle malformed URLs safely.
-   Use reproducible ML training (`random_state` set).
-   Include clear comments and docstrings.
-   Follow professional coding standards.

## Installation Instructions

1.  **Clone the repository (if applicable) or create the project directory:**
    ```bash
    git clone <repository_url>
    cd sc0mdetctor # Or your project directory
    ```

2.  **Create a virtual environment:**
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```
    (On Windows, use `.venv\Scripts\activate`)

4.  **Install required libraries:**
    ```bash
    pip install pandas scikit-learn numpy tldextract requests rich prompt_toolkit
    ```

## Dataset
For demonstration and development purposes, this project uses a large, synthetically generated dataset (`phishing_dataset.csv`). This dataset is created using `dataset/generate_advanced_dataset.py` to simulate a diverse range of legitimate and phishing URLs, incorporating various phishing tactics like typos, suspicious keywords, and IP-based URLs.

**For production use, it is critical to replace this simulated dataset with a real, large-scale, and regularly updated public phishing URL dataset.** Instructions for obtaining such datasets are typically available from research institutions and security organizations.

To regenerate the simulated dataset (e.g., if you modify the generation logic):
```bash
source .venv/bin/activate
python dataset/generate_advanced_dataset.py
```

## Training Instructions

1.  **Ensure your virtual environment is activated.**
2.  **Run the training script:**
    ```bash
    ./train.sh [model_type]
    ```
    -   `[model_type]` is optional. Choose `logistic_regression` (default) or `random_forest`.
    -   Example for Logistic Regression (default): `./train.sh`
    -   Example for Random Forest: `./train.sh random_forest`

    This script will:
    -   Load the dataset (`phishing_dataset.csv`).
    -   Extract the expanded set of features from each URL.
    -   Scale these features using `StandardScaler`.
    -   Split the data into training and testing sets.
    -   Train the specified machine learning model.
    -   Evaluate the model's performance.
    -   Save the trained model (`<model_type>_model.joblib`) and the `StandardScaler` (`scaler.joblib`) into the `model/` directory. The scaler is crucial for consistent feature scaling during detection.

## Explanation of Advanced Feature Engineering (`features/feature_extractor.py`)

The `features/feature_extractor.py` module extracts a comprehensive set of lexical and structural features from URLs, designed to capture subtle indicators of phishing attempts. This includes URL length, dot count, IP address presence, HTTPS usage, special character counts, suspicious keywords, URL entropy, subdomain counts, hostname/path/query/fragment lengths, digit counts, uncommon TLDs, suspicious TLDs, and non-ASCII character counts.

## LLM Integration (Placeholder)

The `get_llm_response` function in `detect.py` currently provides simulated, keyword-based responses. To integrate a real Large Language Model:

1.  **Choose an LLM Provider:** Select a service like Google AI (Gemini API), OpenAI, Anthropic, etc.
2.  **Obtain API Key:** Acquire an API key from your chosen provider. **Never hardcode API keys directly into your code.** Use environment variables for secure storage.
3.  **Replace Placeholder Logic:** Modify the `get_llm_response` function to make actual API calls to your LLM provider. This will involve using the provider's Python client library (e.g., `google.generativeai` for Gemini).
4.  **Customize Prompts:** Design effective prompts for the LLM to guide its responses, potentially including system instructions or few-shot examples for better intent recognition and response quality.

## Example Usage (Interactive Conversational Mode)

To start the Conversational AI Phishing Detector:
```bash
./detect.sh [model_path]
```
-   `[model_path]` is optional. If provided, it specifies the `.joblib` file for the ML model to use (e.g., `model/random_forest_model.joblib`). If omitted, the script will attempt to load `model/random_forest_model.joblib` first, then `model/logistic_regression_model.joblib`.

Once started, the tool will present a welcome message and then continuously prompt you for input. You can chat naturally or paste URLs for analysis. The chat history will be displayed dynamically using `rich.Live`, with history scrolling upwards as new messages arrive.

**Interactive Session Example:**
```
╔═════════════════════════════════════════════════════════════════════════════════════════ Welcome ═════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                                                                                                           ║
║                       ✨ [bold green]Advanced Phishing URL Detector & Chat Assistant[/bold green] ✨                                                                                     ║
║                       Using model: [bold cyan]model/random_forest_model.joblib[/bold cyan]                                                                                                 ║
║                       I'm ready to chat! Ask me anything or paste a URL for analysis.                                                                                                     ║
║                       Type '[bold yellow]exit[/bold yellow]' to quit.                                                                                                                     ║
║                                                                                                                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
(The dynamic chat history will appear below, with new messages scrolling old ones up)

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
🚀 You:> hello
```
(After typing "hello" and pressing Enter, the screen will update dynamically to show:)
```
╔═════════════════════════════════════════════════════════════════════════════════════════ Welcome ═════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                                                                                                           ║
║                       ✨ [bold green]Advanced Phishing URL Detector & Chat Assistant[/bold green] ✨                                                                                     ║
║                       Using model: [bold cyan]model/random_forest_model.joblib[/bold cyan]                                                                                                 ║
║                       I'm ready to chat! Ask me anything or paste a URL for analysis.                                                                                                     ║
║                       Type '[bold yellow]exit[/bold yellow]' to quit.                                                                                                                     ║
║                                                                                                                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
╭─ You ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ hello                                                                                                                                                                                     │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ AI ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Hello there! How can I help you today? Do you have a URL you'd like me to check?                                                                                                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
🚀 You:>
```

## Future Improvements
-   **Real LLM Integration:** Replace the `get_llm_response` placeholder function with actual API calls to a chosen LLM service (e.g., Google Gemini API, OpenAI API) using environment variables for API key management.
-   **Advanced Intent Recognition:** Implement more sophisticated NLP techniques (e.g., keyword extraction, entity recognition, fine-tuned LLM calls) to better understand user queries and dynamically invoke tools (like URL detection).
-   **Context Management:** Implement a mechanism for the LLM to remember past turns in the conversation to provide more coherent and helpful responses.
-   **Custom `prompt_toolkit` Widgets:** For truly advanced chat-like input (e.g., multi-line input boxes, auto-suggestions specific to conversation), further custom `prompt_toolkit` widgets could be explored.
-   **Real Dataset Integration:** Replace the simulated dataset with a continuously updated, real-world phishing dataset for maximum accuracy and generalizability.
-   **Advanced ML Model Evaluation:** Implement more comprehensive cross-validation strategies and hyperparameter tuning for optimal model performance.
-   **More Sophisticated Risk Scoring:** Implement dynamic thresholds for risk levels based on model certainty and potential false positive/negative rates.
-   **Threat Intelligence Integration:** Integrate with services like Google Safe Browsing, PhishTank, or domain age/WHOIS lookups (via external APIs, if allowed by constraints).
-   **Domain Reputation:** Incorporate features derived from domain reputation services.
-   **Content-Based Analysis:** Extend to analyze the webpage itself (e.g., HTML structure, embedded forms, JavaScript) for deeper insights (requires fetching and parsing web content, which has performance and security implications).
-   **User Feedback Loop:** Implement a mechanism for security analysts to provide feedback, which can be used to retrain and improve the ML model over time.
