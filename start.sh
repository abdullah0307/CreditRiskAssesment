#!/bin/bash

echo "Starting application..."
pip freeze
gunicorn app:app