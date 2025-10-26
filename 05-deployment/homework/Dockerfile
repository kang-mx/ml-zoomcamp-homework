FROM agrigorev/zoomcamp-model:2025
# add your stuff here
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked

COPY "predict-v2.py" "pipeline_v1.bin" ./

EXPOSE 3000

ENTRYPOINT ["uv", "run", "uvicorn", "predict-v2:app", "--host", "0.0.0.0", "--port", "3000"]