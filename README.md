# Dagster Portfolio Sample

A sample project demonstrating how to build and validate a simple portfolio using Dagster and vBase. This project creates a daily portfolio that takes a long position in SPY if the price return is positive and a short position if negative.

## Quickstart Guide

1. Clone the repository:
```bash
git clone https://github.com/validityBase/dagster-portfolio-sample.git
cd dagster-portfolio-sample
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```bash
# vBase Configuration
VBASE_API_KEY=your_api_key_here           # API key for vBase authentication
VBASE_API_URL=your_api_url_here           # vBase API endpoint URL
VBASE_COMMITMENT_SERVICE_PRIVATE_KEY=your_private_key_here  # Private key for vBase commitment service

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key     # AWS access key for S3 operations
AWS_SECRET_ACCESS_KEY=your_aws_secret_key # AWS secret key for S3 operations
S3_BUCKET=your_bucket_name                # S3 bucket name for storing portfolio data
S3_FOLDER=your_folder_name                # S3 folder path within the bucket
```

For Dagster Cloud deployment, these variables should be set as secrets in your Dagster Cloud organization settings.

5. Run the Dagster development server:
```bash
dagster dev
```

6. Access the Dagster UI at `http://localhost:3000`

## Architecture

The project consists of several key components:

1. **[Portfolio Producer](dagster_pipelines/assets/portfolio_producer.py)**: Core logic for generating portfolio positions based on SPY price movements. This is the file that will be changed to define new portfolios and datasets.
2. **[Portfolio Asset](dagster_pipelines/assets/portfolio_asset.py)**: Dagster and vBase logic for managing the produced portfolio.

### Portfolio Producer

The portfolio producer is designed to be:
- **Reusable**: Can be called from different contexts (Dagster, standalone scripts, etc.)
- **Testable**: Has clear inputs and outputs
- **Loggable**: Includes comprehensive logging for debugging and monitoring

## Development

- The project uses Pylint for code quality checks (minimum score: 8.0)
- GitHub Actions automatically runs Pylint on all pushes and pull requests
- Pre-commit hooks are configured to run code quality checks before commits

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.txt](LICENSE.txt) file for details.
