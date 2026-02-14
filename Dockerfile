#--- START OF FULL, FINAL, AND CONFIRMED READY-TO-USE FILE: Dockerfile ---
# ================================
# Base Image
# ================================
FROM python:3.11-slim

# ================================
# Environment
# ================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Aden
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# ================================
# System Dependencies
# ================================
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    libatk-bridge2.0-0 \
    libxkbcommon0 \
    libxcomposite1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    tzdata \
    procps \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Timezone Configuration
# ================================
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ================================
# Working Directory
# ================================
WORKDIR /app

# ================================
# Python Dependencies (Cached Layer)
# ================================
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# Install Playwright
# ================================
RUN playwright install chromium --with-deps

# ================================
# Copy Application
# ================================
COPY . /app

# Create directory for evidence
RUN mkdir -p /app/evidence

# ================================
# Healthcheck
# ================================
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ps aux | grep "[p]ython" || exit 1

# ================================
# Run
# ================================
CMD ["python", "-m", "src.main"]
#--- END OF FULL, FINAL, AND CONFIRMED READY-TO-USE FILE: Dockerfile ---