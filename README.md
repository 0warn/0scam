# Advanced AI-based Phishing URL Detection Tool

This project implements an advanced AI-based Phishing URL Detection tool that classifies a given URL as Legitimate, Suspicious, or Phishing. It is designed to be lightweight, accurate, and suitable for SOC / blue-team use, with a focus on future-proofing through extensible feature engineering and robust model choices, now featuring an enhanced and attractive interactive CLI.

## Project Goal
Create a Python-based system that classifies a given URL as:
- **Legitimate**
- **Suspicious**
- **Phishing**

## Functional Requirements
1.  **Accept a URL as input:** Features an attractive and interactive CLI for continuous detection.
2.  **Extract meaningful lexical and structural features from the URL:** This version includes a significantly expanded set of features.
3.  **Train a machine learning model:** Using a large (simulated or real) phishing URL dataset.
4.  **Use simple, explainable models:** Logistic Regression or Random Forest, with emphasis on robust training.
5.  **Output:**
    *   Classification result
    *   Risk level (LOW / MEDIUM / HIGH)
    *   Confidence score (percentage)
    *   **URL Availability and HTTP Status Code:** Checks if the URL is reachable and its network response.

## Technical Constraints
-   **Language:** Python 3
-   **Libraries:** `pandas`, `scikit-learn`, `numpy`, `tldextract`, `requests`, `rich`
-   No deep learning
-   No external API calls for ML features (direct URL availability check is permitted)
-   Code must be modular and clean

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
├── detect.py                        # Updated detection script with interactive mode, URL check, and rich UI fixes
├── detect.sh                        # Updated convenience script for interactive detection
├── train.py                         # Updated training script
├── train.sh                         # Updated convenience script for training
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
    pip install pandas scikit-learn numpy tldextract requests rich
    ```

## Dataset
For demonstration and development purposes, this project uses a large, synthetically generated dataset (`phishing_dataset.csv`). This dataset is created using `dataset/generate_advanced_dataset.py` to simulate a diverse range of legitimate and phishing URLs, incorporating various phishing tactics like typos, suspicious keywords, and IP-based URLs.

**For production use, it is critical to replace this simulated dataset with a real, large-scale, and regularly updated public phishing URL dataset.** Instructions for obtaining such datasets are typically available from research institutions or security organizations.

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

The `features/feature_extractor.py` module now extracts a comprehensive set of lexical and structural features from URLs, designed to capture subtle indicators of phishing attempts:

-   **`url_length`**: Total number of characters in the URL. Very long URLs can sometimes be used for obfuscation.
-   **`num_dots`**: Count of dots (`.`) in the URL. An excessive number can indicate subdomain abuse or attempts to hide the true domain.
-   **`has_ip_address`**: Binary (1/0) indicating if an IP address (IPv4 or IPv6) is used as the hostname. Phishers often use IPs to bypass DNS-based filters.
-   **`uses_https`**: Binary (1/0) indicating if the URL uses HTTPS. While not a definitive sign of legitimacy, its absence or suspicious use is a red flag.
-   **`special_char_count`**: Counts occurrences of various special characters (`@`, `?`, `-`, `=`, `_`, `~`, `&`, `#`, `%`, `.`, `+`, `/`, `\`, `:`) often employed in phishing for obfuscation or to confuse users.
-   **`has_suspicious_keywords`**: Binary (1/0) checking for keywords like `login`, `secure`, `verify`, `update`, `account`, `banking`, `confirm`, `password`, `webscr`, `signin` (case-insensitive). These are common lures in phishing.
-   **`url_entropy`**: Measures the randomness of the URL string using Shannon entropy. High entropy can suggest dynamically generated or obfuscated URLs.
-   **`num_subdomains`**: Counts the number of subdomains. An unusual number of subdomains can be a sign of phishing (e.g., `paypal.com.secure.login.evilsite.com`).
-   **`hostname_length`**: Length of the hostname part of the URL.
-   **`path_length`**: Length of the path part of the URL.
-   **`query_length`**: Length of the query string part of the URL.
-   **`fragment_length`**: Length of the fragment part of the URL.
-   **`digits_in_url`**: Count of numerical digits in the entire URL. High numbers might indicate IP addresses in unexpected places or attempts to mimic legitimate IDs.
-   **`has_uncommon_tld`**: Binary (1/0) indicating if the Top-Level Domain (TLD) is not in a predefined list of common TLDs. Unusual TLDs can be favored by phishers.
-   **`non_ascii_chars`**: Counts non-ASCII characters. Presence of these can indicate homograph attacks (using visually similar characters from different alphabets).

## Example Usage (Interactive Mode with Enhanced UI)

To start the interactive phishing URL detection tool:
```bash
./detect.sh [model_path]
```
-   `[model_path]` is optional. If provided, it specifies the `.joblib` file for the model to use (e.g., `model/random_forest_model.joblib`). If omitted, the script will attempt to load `model/random_forest_model.joblib` first, then `model/logistic_regression_model.joblib`.

Once started, the tool will prompt you to enter URLs continuously until you type `exit`. The output will be presented with rich formatting, colors, icons, and animated status messages for a clear, engaging, and professional user experience.

**Interactive Session Example:**
```
╭────────────────────────────────────────────────────────────────────────────────── Welcome ───────────────────────────────────────────────────────────────────────────────────╮
│                                          ✨ Advanced Phishing URL Detector ✨                                           │
│                 Using model: [bold cyan]model/random_forest_model.joblib[/bold cyan]                                                                                                                       │
│                 Enter a URL to check, or type '[bold yellow]exit[/bold yellow]' to quit.                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
🚀 Enter URL : https://www.google.com
⠏ 🌐 Checking URL availability and status...
⠸ 🧠 Analyzing URL for phishing characteristics...
╭─────────────────────────────── Results for: [bold white]https://www.google.com[/bold white] ────────────────────────────────╮
│                                                                                                                                  │
│     Property      Value                                                                                                            │
│ ───────────────── ──────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ Classification:   ✅ Legitimate                                                                                                   │
│ Risk Level:       🛡️ LOW                                                                                                          │
│ Confidence:       [bold magenta]0.00%[/bold magenta]                                                                               │
│ HTTP Status:      ✅ 200                                                                                                          │
│ Reachability:     ✅ Reachable                                                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
🚀 Enter URL : http://example.com/bad/login
⠏ 🌐 Checking URL availability and status...
⠸ 🧠 Analyzing URL for phishing characteristics...
╭─────────────────────────── Results for: [bold white]http://example.com/bad/login[/bold white] ───────────────────────────╮
│                                                                                                                                  │
│     Property      Value                                                                                                            │
│ ───────────────── ──────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ Classification:   🎣 Phishing                                                                                                     │
│ Risk Level:       ⚠️ MEDIUM                                                                                                       │
│ Confidence:       [bold magenta]55.00%[/bold magenta]                                                                              │
│ HTTP Status:      ✅ 200                                                                                                          │
│ Reachability:     ✅ Reachable                                                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
🚀 Enter URL : http://nonexistent-phishing-site-12345.com
⠏ 🌐 Checking URL availability and status...
⠸ 🧠 Analyzing URL for phishing characteristics...
╭─────────────────────────────── Results for: [bold white]http://nonexistent-phishing-site-12345.com[/bold white] ────────────────────────────────╮
│                                                                                                                                  │
│     Property      Value                                                                                                            │
│ ───────────────── ──────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ Classification:   ✅ Legitimate                                                                                                   │
│ Risk Level:       🛡️ LOW                                                                                                          │
│ Confidence:       [bold magenta]0.00%[/bold magenta]                                                                               │
│ HTTP Status:      ❌ N/A                                                                                                         │
│ Reachability:     ❌ Connection Error (Host Down/Unreachable)                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
🚀 Enter URL : exit
Exiting Phishing URL Detector. Goodbye! 👋
```

## Future Improvements
-   **Real Dataset Integration:** Replace the simulated dataset with a continuously updated, real-world phishing dataset for maximum accuracy and generalizability.
-   **Advanced Model Evaluation:** Implement more comprehensive cross-validation strategies and hyperparameter tuning for optimal model performance.
-   **More Sophisticated Risk Scoring:** Implement dynamic thresholds for risk levels based on model certainty and potential false positive/negative rates.
-   **Threat Intelligence Integration:** While constrained from external API calls in this version, for a truly future-proof system, integration with services like Google Safe Browsing, PhishTank, or domain age/WHOIS lookups would be crucial.
-   **Domain Reputation:** Incorporate features derived from domain reputation services.
-   **Content-Based Analysis:** Extend to analyze the content of the webpage itself (e.g., HTML structure, embedded forms, JavaScript) for deeper insights (requires fetching and parsing web content, which has performance and security implications).
-   **User Feedback Loop:** Implement a mechanism for security analysts to provide feedback, which can be used to retrain and improve the model over time.
