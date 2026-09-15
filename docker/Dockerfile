FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY mcp.json ./
COPY mcp_servers ./mcp_servers
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["hippo"]
CMD ["--help"]
