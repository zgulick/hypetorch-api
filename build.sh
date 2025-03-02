#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies for psycopg2
apt-get update
apt-get install -y libpq-dev python3-dev

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt