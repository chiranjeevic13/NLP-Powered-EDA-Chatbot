# NLP-Powered EDA Chatbot

## Overview
This project implements an **NLP-powered chatbot** that enables users to query datasets using natural language and perform advanced exploratory data analysis (EDA). Built with Python and FastAPI, the chatbot leverages state-of-the-art NLP techniques and a language model to process queries, generate insights, and provide statistical analysis. It is designed for data scientists, analysts, and non-technical users who want to explore datasets intuitively via a web API or Jupyter notebook.

The project integrates:
- **Advanced EDA**: Statistical analysis, outlier detection, distribution analysis, and feature engineering recommendations.
- **NLP Query Processing**: Natural language understanding using `spaCy` and fuzzy matching for column identification.
- **LLM Integration**: A language model (`facebook/opt-350m`) for handling complex queries and generating contextual responses.
- **FastAPI Web Interface**: RESTful API for dataset upload, EDA report generation, and query processing.

This repository demonstrates expertise in **AI**, **machine learning**, **NLP**, **data science**.

## Features
- **Comprehensive EDA Module**:
  - Analyzes numerical and categorical columns with metrics like skewness, kurtosis, entropy, and quantiles.
  - Detects outliers using the IQR method and provides scalability insights (e.g., memory usage, sparsity).
  - Generates feature engineering recommendations for high-variability columns.
  - Saves insights in multiple formats (JSON, YAML, CSV, TXT).
- **Mathematical Query Module**:
  - Processes natural language queries for statistical operations (e.g., mean, median, standard deviation, min/max).
  - Uses advanced NLP with `spaCy` for column matching and semantic understanding.
  - Supports filtering queries with conditions (e.g., "salary > 50000").
- **Data Query Module**:
  - Handles top-N, extreme value, and group-by queries (e.g., "top 5 salaries", "group by department").
  - Employs fuzzy matching and regex for robust query parsing.
- **LLM-Powered Open QA**:
  - Integrates `facebook/opt-350m` for answering open-ended questions about the dataset.
  - Generates context-aware responses based on dataset characteristics.
- **FastAPI Endpoints**:
  - Upload CSV datasets and manage sessions.
  - Generate and download EDA reports in multiple formats.
  - Process natural language queries via a RESTful API.
- **Modular Design**:
  - Organized into reusable classes (`EDAModule`, `UltraAdvancedMathModule`, `AdvancedDataQueryModule`, `OpenQAModule`).
  - Includes logging and error handling for production-grade reliability.

## Technologies Used
- **Python Libraries**: `pandas`, `numpy`, `fastapi`, `uvicorn`, `transformers`, `torch`, `spacy`, `fuzzywuzzy`, `scipy`, `yaml`, `python-multipart`
- **NLP Tools**: `spaCy` (en_core_web_sm), Hugging Face Transformers
- **Language Model**: `facebook/opt-350m`
- **Web Framework**: FastAPI
- **Environment**: Python 3.8+, GPU support (optional, CUDA recommended)

## Installation
Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chiranjeevic13/EDA-Chatbot.git
   cd EDA-Chatbot
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Prepare Dataset**:
   - Use your own CSV dataset with numerical and categorical columns.
   - Note: The repository does not include a sample dataset. You can create one (e.g., `data/salary_data.csv`) with columns like `YearsExperience`, `Age`, and `Salary`.

6. **GPU Setup (Optional)**:
   - Ensure CUDA is installed for GPU acceleration with PyTorch.
   - Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

## Usage

### Option 1: Run the FastAPI Application
1. **Start the FastAPI Server**:
   ```bash
   uvicorn app.main:app --reload
   ```
   - Access the API at `http://localhost:8000`.
   - View interactive API documentation at `http://localhost:8000/docs`.

2. **API Endpoints**:
   - **Upload Dataset**:
     ```bash
     curl -X POST "http://localhost:8000/upload_dataset" -F "file=@path/to/your/dataset.csv"
     ```
     Response:
     ```json
     {"session_id": "123e4567-e89b-12d3-a456-426614174000", "message": "Dataset uploaded successfully"}
     ```
   - **Generate EDA Report**:
     ```bash
     curl -X POST "http://localhost:8000/generate_eda/123e4567-e89b-12d3-a456-426614174000" \
          -H "Content-Type: application/json" -d '{"formats": ["json", "txt"]}'
     ```
     Response:
     ```json
     {"message": "EDA report generated", "report_directory": "eda_reports/20250615_143022"}
     ```
   - **Download EDA Report**:
     ```bash
     curl -X GET "http://localhost:8000/download_eda/123e4567-e89b-12d3-a456-426614174000/json" \
          -o eda_report.json
     ```
   - **Process Query**:
     ```bash
     curl -X POST "http://localhost:8000/query/123e4567-e89b-12d3-a456-426614174000" \
          -H "Content-Type: application/json" -d '{"query": "What is the average salary?"}'
     ```
     Response (example):
     ```json
     {"query": "What is the average salary?", "response": {"mean": 76003.0}}
     ```
   - **Health Check**:
     ```bash
     curl http://localhost:8000/health
     ```
     Response:
     ```json
     {"status": "healthy"}
     ```

3. **Example Queries**:
   - Statistical: "Calculate the mean salary", "What is the maximum age?"
   - Data Selection: "Show top 5 employees by salary", "Group by department and show average experience"
   - Filtering: "Filter employees where salary > 50000"
   - Open-Ended: "Tell me about the dataset", "What insights can you provide about salaries?"

### Option 2: Use the Jupyter Notebook
1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook "EDA Chatbot Using Transformers.ipynb"
   ```

2. **Load Your Dataset**:
   ```python
   import pandas as pd
   df = pd.read_csv('path/to/your/dataset.csv')
   ```

3. **Run the Chatbot**:
   ```python
   from app.modules.opena_module import OpenQAModule
   qa_module = OpenQAModule(df)
   query = "What is the average salary?"
   response = qa_module.process_query(query)
   print(response)
   ```

4. **Generate EDA Report**:
   ```python
   from app.modules.eda_module import EDAModule
   eda = EDAModule(df)
   eda.run_and_save(formats=['json', 'yaml', 'txt'])
   ```

## Example Output
**API Query**:
```bash
curl -X POST "http://localhost:8000/query/123e4567-e89b-12d3-a456-426614174000" \
     -H "Content-Type: application/json" -d '{"query": "What is the minimum salary?"}'
```
**Response**:
```json
{
  "query": "What is the minimum salary?",
  "response": {"mean": 76003.0, "std": 27414.43, "min": 37731.0, "max": 122391.0, "count": 30}
}
```

**EDA Report Snippet** (from `eda_reports/eda_insights_*.txt`):
```
Scalability Analysis:
- Total Memory Usage: 0.02 MB
- Total Rows: 30
- Total Columns: 3
- Data Sparsity Ratio: 0.00%

Detailed Statistical Analysis:
Numeric Columns:
- Salary: {'skewness': 0.35, 'kurtosis': -0.81, ...}
```

## Project Structure
```
EDA-Chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── eda_module.py       # EDAModule class
│   │   ├── math_module.py      # UltraAdvancedMathModule class
│   │   ├── query_module.py     # AdvancedDataQueryModule class
│   │   ├── openqa_module.py    # OpenQAModule class
├── EDA Chatbot Using Transformers.ipynb  # Original Jupyter notebook
├── eda_reports/                # Generated EDA reports
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
```

## Future Improvements
- Add visualizations (e.g., histograms, correlation heatmaps) to the EDA module using `matplotlib` or `seaborn`.
- Upgrade to a more powerful LLM (e.g., Qwen2-VL-2B-Instruct) for improved query handling.
- Implement a database (e.g., MongoDB) for persistent session management.
- Add authentication and rate limiting for API security.
- Support larger datasets with optimized memory usage and parallel processing.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## Contact
- **Author**: Chiranjeevi C
- **Email**: chiruc1305@gmail.com
- **LinkedIn**: [Chiranjeevi C](https://www.linkedin.com/in/chiranjeevi-c-1305c/)
- **GitHub**: [chiranjeevic13](https://github.com/chiranjeevic13)

Feel free to reach out with questions or feedback!
