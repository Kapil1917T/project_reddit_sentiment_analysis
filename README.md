# Reddit Stock Sentiment Analysis Dashboard 📊

A real-time dashboard that analyzes sentiment of stock-related discussions on Reddit using Natural Language Processing and Machine Learning.

Dashboard Link - http://192.168.1.104:8504/

## 🌟 Features

- **Multi-Source Data Collection**
  - Fetches posts from multiple subreddits (r/wallstreetbets, r/stocks, r/investing)
  - Configurable time range and post score filtering
  - Real-time data gathering

- **Advanced Sentiment Analysis**
  - Utilizes FinBERT (Financial BERT) for domain-specific sentiment analysis
  - Provides sentiment scores and confidence metrics
  - Handles both text and title analysis

- **Comparative Analysis**
  - Cross-stock sentiment comparison
  - Competitor analysis
  - Volume-sentiment correlation

- **Interactive Visualization**
  - Sentiment distribution charts
  - Confidence score analysis
  - Temporal sentiment patterns
  - Post volume correlation
  - Most influential posts tracking

## 🛠️ Technology Stack

- **Data Collection**: PRAW (Python Reddit API Wrapper)
- **Text Processing**: NLTK, spaCy
- **Sentiment Analysis**: PyTorch, Transformers (FinBERT)
- **Visualization**: Streamlit, Plotly
- **Data Processing**: Pandas, NumPy
- **Version Control**: Git, GitHub

## 📋 Prerequisites

- Python 3.8+
- Reddit API credentials
- Required Python packages (see requirements.txt)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/reddit-sentiment-analysis.git
   cd reddit-sentiment-analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # For Windows
   venv\Scripts\activate
   # For macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a `.env` file in the project root
   ```env
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

## 💻 Usage

1. **Start the dashboard**
   ```bash
   streamlit run src/visualization/dashboard.py
   ```

2. **Using the dashboard**
   - Enter a stock symbol (e.g., AAPL, MSFT)
   - Select competitor stocks for comparison
   - Choose subreddits to analyze
   - Adjust time range and minimum post score
   - Click "Run Analysis"

## 📊 Dashboard Features

1. **Main Metrics**
   - Total post count
   - Sentiment distribution
   - Average sentiment scores
   - Confidence metrics

2. **Visualizations**
   - Sentiment distribution pie chart
   - Confidence score histogram
   - Volume-sentiment correlation
   - Competitor comparison chart
   - Temporal sentiment heatmap

3. **Detailed Analysis**
   - Most positive/negative posts
   - High-impact posts
   - Raw data access
   - CSV export functionality

## 🏗️ Project Structure

```
project_reddit_sentiment_analysis/
├── src/
│   ├── data/
│   │   └── reddit_scraper.py
│   ├── models/
│   │   ├── bert_model.py
│   │   └── sentiment_analyzer.py
│   ├── preprocessing/
│   │   └── text_processor.py
│   └── visualization/
│       └── dashboard.py
├── tests/
│   ├── test_reddit_scraper.py
│   └── test_sentiment_analyzer.py
├── .env
├── requirements.txt
└── README.md
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📈 Sample Results

The dashboard provides:
- Real-time sentiment analysis
- Comparative stock analysis
- Temporal pattern recognition
- Confidence scoring
- Detailed post analysis

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FinBERT model from ProsusAI
- Reddit API and PRAW library
- Streamlit community
- All contributors and testers

## 📧 Contact

Name - Kapil Tare

Email - kmtare@syr.edu

Project Link: [[https://github.com/Kapil1917T/reddit-sentiment-analysis](https://github.com/Kapil1917T/reddit-sentiment-analysis)](https://github.com/Kapil1917T/project_reddit_sentiment_analysis/tree/main)

## 🚧 Future Improvements

- Real-time price correlation
- Additional data sources integration
- Advanced technical analysis
- Historical trend analysis
- Machine learning model improvements
