# Test-judge image: Python 3.12 + harness deps. trixie (glibc 2.41) runs ai_processor built on the host.
FROM python:3.12-slim-trixie
RUN pip install --no-cache-dir allure-pytest assertpy clickhouse-connect clickhouse-driver h5py logger matplotlib \
    numpy onnxruntime pandas psycopg2-binary "psycopg[binary]" pytest pytest-check pytest-dotenv python-dotenv \
    reportlab requests scikit-learn seaborn tabulate tenacity
