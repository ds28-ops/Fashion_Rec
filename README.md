# Fashion Recommendation Agent 🧵

A powerful fashion recommendation system that uses semantic search with vector embeddings and AI-powered chat interface to help users find relevant clothing items from a local dataset.

## Features

- **Semantic Search**: Uses OpenAI's text-embedding-3-large model to find clothing items based on natural language queries
- **AI-Powered Chat**: Interactive chat interface powered by GPT-4o for personalized recommendations
- **Visual Results**: Display matching clothing items with images in an intuitive grid layout
- **FAISS Vector Search**: High-performance similarity search using Facebook's FAISS library
- **Streamlit Web Interface**: Clean, user-friendly web application

## Demo

The application provides:
- Natural language search (e.g., "I'm looking for a white formal t-shirt")
- Top-K configurable results (1-12 items)
- Image gallery with detailed item descriptions
- AI assistant that explains differences between similar items

## Prerequisites

- Python 3.8+
- OpenAI API key
- Fashion dataset with images

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Fashion_Rec.git
   cd Fashion_Rec
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   IMAGES_DIR=data
   INDEX_PATH=index.faiss
   META_PATH=meta.parquet
   ```

## Dataset Setup

1. **Prepare your dataset**
   
   Your CSV file should have these columns:
   - `image`: filename of the product image
   - `display name`: product display name
   - `description`: detailed product description  
   - `category`: product category (e.g., "Tshirts", "Jeans", "Shoes")

2. **Add your images**
   
   Place all product images in the `data/` folder. The CSV should reference these image filenames.

3. **Build the search index**
   
   This step creates the FAISS vector index and metadata:
   ```bash
   python build_index.py --csv data.csv --images_dir data --out_index index.faiss --out_meta meta.parquet
   ```
   
   Optional parameters:
   - `--limit N`: Process only first N items (useful for testing)
   - `--csv path/to/your/data.csv`: Custom CSV file path

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Search for items** using natural language:
   - "Show me black sports shoes"
   - "I need a formal shirt for office"
   - "Looking for summer dresses under $50"

## Project Structure

```
Fashion_Rec/
├── app.py              # Main Streamlit application
├── build_index.py      # Script to build FAISS index from dataset
├── data.csv            # Fashion dataset (sample included)
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── .gitignore         # Git ignore file
└── README.md          # This file

# Generated files (created by build_index.py):
├── index.faiss        # FAISS vector index
└── meta.parquet       # Product metadata
```

## Technology Stack

- **Frontend**: Streamlit
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: OpenAI text-embedding-3-large
- **Chat AI**: OpenAI GPT-4o
- **Data Processing**: Pandas, NumPy
- **LangChain**: For chat message handling

## Configuration

You can customize the application by modifying these settings in `app.py`:

- **Embedding Model**: Change `text-embedding-3-large` to other OpenAI models
- **Chat Model**: Modify `gpt-4o` in the LLM initialization
- **Search Results**: Adjust the default `k` value for top-K results
- **System Prompt**: Customize the AI assistant's personality and behavior

## API Costs

This application uses OpenAI APIs:
- **Embeddings**: ~$0.00013 per 1K tokens (build_index.py)
- **Chat**: ~$0.0025 per 1K input tokens, ~$0.01 per 1K output tokens (app.py)

## Performance Tips

1. **Batch Processing**: `build_index.py` processes embeddings in batches of 64 for efficiency
2. **Caching**: Streamlit caches the FAISS index and LLM model to avoid reloading
3. **Index Size**: For large datasets (>100K items), consider using FAISS IVF indices

## Troubleshooting

**Missing index.faiss or meta.parquet error**:
```bash
python build_index.py
```

**OpenAI API errors**:
- Verify your API key in `.env`
- Check your OpenAI account has sufficient credits

**Image not found warnings**:
- Ensure all images referenced in CSV exist in the images directory
- Check file extensions match exactly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Feel free to use and modify as needed.

## Acknowledgments

- OpenAI for embeddings and chat models
- Facebook Research for FAISS library
- Streamlit for the web framework
- The fashion dataset contributors