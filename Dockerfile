# python image
FROM python:3.12-slim-bullseye

# install pipenv
RUN pip install pipenv && \
    echo "Pipenv installed successfully"

# set working directory
WORKDIR /app

# copy only dependency files first
COPY Pipfile Pipfile.lock ./
RUN echo "Dependency files copied successfully"

# install dependencies
RUN pipenv install --system --deploy --verbose && \
    echo "Dependencies installed successfully"

# copy application code
COPY ./models ./models
COPY ./scripts ./scripts
RUN echo "Application code copied successfully"

# expose port
EXPOSE 9696

# run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "scripts.predict:app"]