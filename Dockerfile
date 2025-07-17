# Use official Streamlit image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port & run
EXPOSE 8501
CMD ["streamlit", "run", "app_stock_forecast.py", "--server.port=8501", "--server.address=0.0.0.0"]
