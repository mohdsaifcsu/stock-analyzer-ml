# Base image
FROM python:3.11

# Set working directory
WORKDIR /app
COPY . /app

# Copy files
COPY requirements.txt .
COPY app_stock_forecast.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app_stock_forecast.py", "--server.port=8501", "--server.address=0.0.0.0"]
