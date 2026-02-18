import re
from urllib.parse import urlparse, urlunparse
import math
import tldextract

# List of common/legitimate TLDs for 'has_uncommon_tld' check
COMMON_TLDS = {
    'com', 'org', 'net', 'gov', 'edu', 'int', 'mil', 'arpa', 'biz', 'info',
    'name', 'pro', 'aero', 'coop', 'museum', 'mobi', 'asia', 'cat', 'tel',
    'jobs', 'travel', 'xxx', 'ch', 'de', 'fr', 'uk', 'us', 'ca', 'au', 'jp',
    'cn', 'in', 'br', 'ru', 'mx', 'za', 'sg', 'th', 'ph', 'id', 'my', 'kr',
    'io', 'dev', 'app', 'online', 'xyz', 'site', 'shop', # Added some more common new gTLDs to avoid false positives
}

# List of TLDs often associated with phishing/malware
SUSPICIOUS_TLDS = {
    'xyz', 'top', 'tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'ru', 'cn', 'info',
    'loan', 'bid', 'download', 'win', 'party', 'science', 'gdn', 'pro',
    'asia', # Some TLDs can be both common and suspicious depending on context, we choose to flag them if often abused
    # '.corn' would be considered unknown and thus potentially suspicious by default if not in common
}


def get_url_length(url):
    """Returns the length of the URL."""
    return len(url)

def get_num_dots(url):
    """Returns the number of dots in the URL."""
    return url.count('.')

def has_ip_address(url):
    """Checks if the URL uses an IP address instead of a domain name."""
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if hostname:
        # Regex for IPv4 address
        ipv4_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
        # Regex for IPv6 address (simplified for common formats)
        ipv6_pattern = r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$" # Simple, covers full form
        if re.match(ipv4_pattern, hostname) or re.match(ipv6_pattern, hostname):
            return True
        # Also check for hex-encoded IPs (e.g., 0x7f000001) - complex, for advanced.
        # For this scope, stick to standard dotted-decimal/colon-separated.
    return False

def uses_https(url):
    """Checks if the URL uses HTTPS."""
    return urlparse(url).scheme == 'https'

def count_special_characters(url):
    """Counts the occurrences of specified special characters in the URL."""
    special_chars = ['@', '?', '-', '=', '_', '~', '&', '#', '%', '.', '+', '/', '\\', ':']
    count = 0
    for char in special_chars:
        count += url.count(char)
    return count

def has_suspicious_keywords(url):
    """Checks for the presence of suspicious keywords in the URL."""
    keywords = ['login', 'secure', 'verify', 'update', 'account', 'banking', 'confirm', 'password', 'webscr', 'signin']
    url_lower = url.lower()
    for keyword in keywords:
        if keyword in url_lower:
            return 1
    return 0

def get_url_entropy(url):
    """Calculates the Shannon entropy of the URL string."""
    if not url:
        return 0.0
    freq = {}
    for char in url:
        freq[char] = freq.get(char, 0) + 1
    entropy = 0.0
    total_chars = len(url)
    for char_freq in freq.values():
        probability = char_freq / total_chars
        entropy -= probability * math.log2(probability)
    return entropy

def get_num_subdomains(url):
    """Returns the number of subdomains."""
    ext = tldextract.extract(url)
    if ext.subdomain:
        return len(ext.subdomain.split('.'))
    return 0

def get_hostname_length(url):
    """Returns the length of the hostname."""
    return len(urlparse(url).hostname) if urlparse(url).hostname else 0

def get_path_length(url):
    """Returns the length of the path."""
    return len(urlparse(url).path)

def get_query_length(url):
    """Returns the length of the query string."""
    return len(urlparse(url).query)

def get_fragment_length(url):
    """Returns the length of the fragment."""
    return len(urlparse(url).fragment)

def count_digits_in_url(url):
    """Counts the number of digits in the URL."""
    return sum(c.isdigit() for c in url)

def has_uncommon_tld(url):
    """Checks if the URL uses a Top-Level Domain (TLD) that is not in a common list."""
    ext = tldextract.extract(url)
    # If the suffix is empty, tldextract couldn't parse it well, which is often suspicious
    if not ext.suffix:
        return 1 # Treat unparseable/missing TLD as uncommon
    return 1 if ext.suffix not in COMMON_TLDS else 0

def is_suspicious_tld(url):
    """Checks if the TLD is in a list of commonly abused/suspicious TLDs."""
    ext = tldextract.extract(url)
    if ext.suffix and ext.suffix in SUSPICIOUS_TLDS:
        return 1
    # Additionally, if tldextract couldn't find a valid suffix, it's also suspicious
    if not ext.suffix and urlparse(url).netloc: # Only if netloc exists but no suffix
        return 1
    return 0

def count_non_ascii_chars(url):
    """Counts the number of non-ASCII characters in the URL."""
    return sum(1 for char in url if ord(char) > 127)

def extract_features(url):
    """
    Extracts a set of lexical and structural features from a given URL.

    Args:
        url (str): The URL to extract features from.

    Returns:
        dict: A dictionary of extracted features.
    """
    # Ensure all URLs are handled, even if malformed, to extract what's possible
    try:
        features = {
            'url_length': get_url_length(url),
            'num_dots': get_num_dots(url),
            'has_ip_address': 1 if has_ip_address(url) else 0,
            'uses_https': 1 if uses_https(url) else 0,
            'special_char_count': count_special_characters(url),
            'has_suspicious_keywords': has_suspicious_keywords(url),
            'url_entropy': get_url_entropy(url),
            'num_subdomains': get_num_subdomains(url),
            'hostname_length': get_hostname_length(url),
            'path_length': get_path_length(url),
            'query_length': get_query_length(url),
            'fragment_length': get_fragment_length(url),
            'digits_in_url': count_digits_in_url(url),
            'has_uncommon_tld': has_uncommon_tld(url),
            'is_suspicious_tld': is_suspicious_tld(url), # New feature
            'non_ascii_chars': count_non_ascii_chars(url)
        }
    except Exception as e:
        # Log the error or return a default feature set if parsing completely fails
        print(f"Error extracting features from URL '{url}': {e}")
        # Return a feature set that might lean towards suspicious or has default values
        default_features = {
            'url_length': len(url), 'num_dots': 0, 'has_ip_address': 0, 'uses_https': 0,
            'special_char_count': 0, 'has_suspicious_keywords': 0, 'url_entropy': 0,
            'num_subdomains': 0, 'hostname_length': 0, 'path_length': 0,
            'query_length': 0, 'fragment_length': 0, 'digits_in_url': 0,
            'has_uncommon_tld': 1, 'is_suspicious_tld': 1, 'non_ascii_chars': 0
        }
        return default_features
    return features

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "https://www.google.com/search?q=hello&ie=UTF-8#top",
        "http://192.168.1.1/admin/login.php?user=root&pass=toor",
        "https://secure-banking.com/verify?account=123",
        "http://phish.com/@user_login_update",
        "https://www.paypal.com-secure-login.ru/webscr/login",
        "http://xn--80ahd1ab2j.xn--p1ai/path/to/page", # Punycode example
        "https://amaz0n.co/signin?ref_=nav_em__signin",
        "https://glthub.corn", # Our problematic URL
        "http://evilsite.xyz/login.php", # Suspicious TLD
        "http://.gq/login" # Malformed/missing TLD
    ]

    # Test tldextract for a domain
    print(tldextract.extract("https://www.google.com/search"))
    print(tldextract.extract("https://mail.google.com"))
    print(tldextract.extract("http://[::1]/path"))
    print(f"TLD for glthub.corn: {tldextract.extract('https://glthub.corn').suffix}")


    for u in test_urls:
        print(f"\nURL: {u}")
        feats = extract_features(u)
        for key, value in feats.items():
            print(f"  {key}: {value}")
        print("-" * 30)
