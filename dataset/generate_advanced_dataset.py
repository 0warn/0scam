import random
from urllib.parse import quote
import pandas as pd

def generate_legitimate_url(base_domain):
    protocols = ["http", "https"]
    tlds = [".com", ".org", ".net", ".io", ".dev"]
    subdomains = ["www", "shop", "blog", "mail", "secure", "app"]
    paths = ["", "/products", "/category/item", "/user/profile", "/articles/news"]
    queries = ["", "?id=123", "?search=query&page=2", "?session_id=abcxyz"]
    fragments = ["", "#section1", "#top"]

    protocol = random.choice(protocols)
    tld = random.choice(tlds)
    subdomain = random.choice(subdomains) if random.random() > 0.3 else "" # Sometimes no subdomain

    domain_part = f"{subdomain}.{base_domain}{tld}" if subdomain else f"{base_domain}{tld}"
    path = random.choice(paths)
    query = random.choice(queries)
    fragment = random.choice(fragments)

    return f"{protocol}://{domain_part}{path}{query}{fragment}"

def generate_phishing_url():
    protocols = ["http", "https"]
    phishing_domains = [
        "login-secure.com", "verify-account.net", "update-banking.org",
        "paypal-safe.info", "amazon-support.co", "microsoft-login.ru",
        "appleid-verify.xyz", "google-security.pw", "fb-help.tk"
    ]
    typo_domains = {
        "paypal.com": ["paypa1.com", "paypai.com", "paypol.com"],
        "google.com": ["gooogle.com", "gogle.com", "googie.com"],
        "amazon.com": ["amaz0n.com", "amaxon.com", "amazn.com"],
        "microsoft.com": ["micr0soft.com", "microsft.com"],
        "apple.com": ["appe.com", "appie.com"]
    }
    keywords = ['login', 'secure', 'verify', 'update', 'account', 'banking', 'confirm', 'password', 'webscr']
    special_chars_paths = ["@login", "@secure", "/_login_", "/-update-", "/login//", "/verify.php"]
    ip_addresses = [
        "http://192.168.1.1/login", "http://10.0.0.5/verify",
        "http://172.16.0.10/update", "http://123.45.67.89/account"
    ]

    choice = random.randint(0, 5) # Increase diversity of phishing tactics

    if choice == 0: # IP address based
        return random.choice(ip_addresses)
    elif choice == 1: # Typo domain
        original_domain = random.choice(list(typo_domains.keys()))
        typo = random.choice(typo_domains[original_domain])
        protocol = random.choice(protocols)
        keyword_path = random.choice(keywords) if random.random() > 0.5 else ""
        return f"{protocol}://{typo}/{keyword_path}"
    elif choice == 2: # Subdomain phishing
        domain = random.choice(phishing_domains)
        legit_part = random.choice(["paypal.com", "google.com", "amazon.com"])
        protocol = random.choice(protocols)
        return f"{protocol}://{legit_part}.{domain}/login"
    elif choice == 3: # Path obfuscation with special chars and keywords
        protocol = random.choice(protocols)
        domain = random.choice(["trusted-site.com", "legit-looking.net"])
        path = random.choice(special_chars_paths)
        keyword = random.choice(keywords)
        return f"{protocol}://{domain}{path}/{keyword}"
    elif choice == 4: # Long and complex URL
        protocol = random.choice(protocols)
        domain = random.choice(phishing_domains)
        num_subdomains = random.randint(2, 5)
        subdomains = ".".join([random.choice(["secure", "login", "web", "docs"]) for _ in range(num_subdomains)])
        path = "/" + "/".join([random.choice(keywords) for _ in range(random.randint(1, 3))])
        query = "?" + "&".join([f"{random.choice(['id','user','token'])}={random.randint(1000,9999)}" for _ in range(random.randint(1,2))])
        return f"{protocol}://{subdomains}.{domain}{path}{query}"
    else: # Direct phishing domain with keywords
        protocol = random.choice(protocols)
        domain = random.choice(phishing_domains)
        keyword = random.choice(keywords)
        return f"{protocol}://{domain}/{keyword}.html"


def generate_dataset(num_legitimate=5000, num_phishing=5000, filename='dataset/phishing_dataset.csv'):
    urls = []
    labels = []

    print(f"Generating {num_legitimate} legitimate URLs...")
    common_domains = ["google", "facebook", "amazon", "microsoft", "apple", "ebay", "wikipedia", "youtube"]
    for _ in range(num_legitimate):
        domain = random.choice(common_domains)
        urls.append(generate_legitimate_url(domain))
        labels.append("legitimate")

    print(f"Generating {num_phishing} phishing URLs...")
    for _ in range(num_phishing):
        urls.append(generate_phishing_url())
        labels.append("phishing")

    df = pd.DataFrame({'url': urls, 'label': labels})
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle
    df.to_csv(filename, index=False)
    print(f"Generated dataset with {len(df)} entries saved to {filename}")

if __name__ == '__main__':
    generate_dataset(num_legitimate=10000, num_phishing=10000)
