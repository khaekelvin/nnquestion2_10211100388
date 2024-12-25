echo "Starting Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT \
         --workers 2 \
         --timeout 120 \
         --log-level debug \
         app:app