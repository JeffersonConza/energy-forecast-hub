# 1. Base image: Use a lightweight Python version
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies first (for faster re-builds)
COPY requirements.txt .

# 4. Install libraries
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Default command (will be overridden by docker-compose)
CMD ["bash"]