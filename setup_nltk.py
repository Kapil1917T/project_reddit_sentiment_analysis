# setup_nltk.py

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download all required NLTK data"""
    required_packages = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    
    for package in required_packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")

if __name__ == "__main__":
    print("Starting NLTK downloads...")
    download_nltk_data()
    print("Completed NLTK setup")