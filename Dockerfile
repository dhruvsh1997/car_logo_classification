# Step 1: Use an official Python image as the base
FROM python:3.10-slim

# Step 2: Set environment variables for Django
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=car_logo_classification.settings

# Step 3: Set the working directory in the container
WORKDIR /app

# Step 4: Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the project files to the working directory in the container
COPY . /app/

# Step 6: Collect static files (for production use)
RUN python manage.py collectstatic --noinput

# Step 7: Expose the port the app runs on
EXPOSE 8000

# Step 8: Set the command to run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
