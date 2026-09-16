# hippo: long-term-memory CLI coding agent.
#   docker build -t hippo -f docker/Dockerfile .
#   docker run --rm -it --env-file .env -v "$PWD:/workspace" -v hippo-data:/data hippo run "task"
#
# Python image + Node runtime (the filesystem MCP server is an npm package) + git.
FROM python:3.12-slim

# Node 20 copied from the official image: smaller and newer than Debian's nodejs package.
COPY --from=node:20-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:20-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE mcp.json ./
COPY src ./src
COPY mcp_servers ./mcp_servers
RUN pip install --no-cache-dir -e . \
 # warm the npx cache so the first `hippo run` does not hit the network for the MCP server
 && (npx -y @modelcontextprotocol/server-filesystem >/dev/null 2>&1 || true)

# The repo you want hippo to work on is mounted at /workspace; memory/traces live in /data.
ENV HIPPO_WORKSPACE=/workspace \
    HIPPO_DATA_DIR=/data \
    HIPPO_MEMORY_BACKEND=auto \
    PYTHONUNBUFFERED=1
RUN mkdir -p /workspace /data
WORKDIR /workspace
VOLUME ["/data"]

ENTRYPOINT ["hippo"]
CMD ["--help"]
