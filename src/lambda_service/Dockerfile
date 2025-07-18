FROM public.ecr.aws/lambda/python:3.12

# Upgrade pip
RUN pip install -U pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY lambda_function.py .
COPY model.py .

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]
