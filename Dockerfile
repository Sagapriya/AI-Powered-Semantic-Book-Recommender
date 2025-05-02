# 1. Use a base Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only the requirements file first (to leverage Docker cache)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# 6. Copy the rest of the project files into the container
COPY . .

# 7. Expose the port that Streamlit will run on
EXPOSE 8501

# 8. Define the command to run your Streamlit app
CMD ["streamlit", "run", "streamlit-dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
