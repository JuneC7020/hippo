"""Generate benchmarks/catalog/synthetic_tools.json from the compact spec below.

The catalog is what `hippo bench` binds: ~180 tools across 15 fictional-but-realistic
servers in MCP `{name, description, inputSchema}` shape, including nested object/array
parameters so schema token costs look like real MCP servers (see `json_schema_to_model`).
Every tool has a canned result, so a benchmark run never touches a real service.

    python scripts/gen_synthetic_catalog.py            # rewrite the JSON
    python scripts/gen_synthetic_catalog.py --check    # exit 1 if the JSON is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "catalog" / "synthetic_tools.json"

# --------------------------------------------------------------------------- mini DSL
# P(name, type, description, required=False)
#   type: str | int | num | bool | str[] | int[] | obj | a full JSON-schema dict for nested shapes

_TYPES = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "num": {"type": "number"},
    "bool": {"type": "boolean"},
    "str[]": {"type": "array", "items": {"type": "string"}},
    "int[]": {"type": "array", "items": {"type": "integer"}},
    "obj": {"type": "object", "additionalProperties": True},
}


def P(name: str, typ: str | dict[str, Any], desc: str, req: bool = False) -> dict[str, Any]:  # noqa: N802
    schema = dict(_TYPES[typ]) if isinstance(typ, str) else dict(typ)
    schema["description"] = desc
    return {"name": name, "schema": schema, "required": req}


def obj(props: dict[str, tuple[str, str]], required: list[str] | None = None) -> dict[str, Any]:
    """Nested object: {field: (type, description)}."""
    return {
        "type": "object",
        "properties": {k: {**_TYPES[t], "description": d} for k, (t, d) in props.items()},
        "required": required or [],
    }


def arr(item: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": item}


def T(name: str, desc: str, params: list[dict[str, Any]], result: str | None = None) -> dict:  # noqa: N802
    return {"name": name, "description": desc, "params": params, "result": result}


# --------------------------------------------------------------------------- servers

SERVERS: dict[str, tuple[str, list[dict[str, Any]]]] = {}

SERVERS["github"] = (
    "GitHub repositories, issues, pull requests, files and Actions.",
    [
        T(
            "create_issue",
            "Create a new issue in a GitHub repository. Returns the issue number and URL.",
            [
                P("repo", "str", "Repository in owner/name form, e.g. acme/webapp", True),
                P("title", "str", "Issue title", True),
                P("body", "str", "Issue body in Markdown"),
                P("labels", "str[]", "Label names to apply"),
                P("assignees", "str[]", "GitHub logins to assign"),
            ],
            '{"number": 418, "url": "https://github.com/acme/webapp/issues/418", "state": "open"}',
        ),
        T(
            "list_issues",
            "List issues in a repository, filtered by state, labels or assignee.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("state", "str", "open, closed or all (default open)"),
                P("labels", "str[]", "Only issues with all of these labels"),
                P("assignee", "str", "Only issues assigned to this login"),
                P("limit", "int", "Maximum number of issues to return (default 30)"),
            ],
            '[{"number": 401, "title": "Flaky checkout test", "state": "open"}, '
            '{"number": 396, "title": "Upgrade to Node 22", "state": "open"}]',
        ),
        T(
            "get_issue",
            "Get one issue by number, including body, labels, assignees and comment count.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("number", "int", "Issue number", True),
            ],
        ),
        T(
            "add_issue_comment",
            "Add a comment to an existing issue or pull request.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("number", "int", "Issue or pull request number", True),
                P("body", "str", "Comment body in Markdown", True),
            ],
        ),
        T(
            "create_pull_request",
            "Open a pull request from head branch into base branch.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("title", "str", "Pull request title", True),
                P("head", "str", "Branch with the changes", True),
                P("base", "str", "Branch to merge into, e.g. main", True),
                P("body", "str", "Description in Markdown"),
                P("draft", "bool", "Open as a draft pull request"),
            ],
        ),
        T(
            "list_pull_requests",
            "List pull requests in a repository by state and base branch.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("state", "str", "open, closed or all"),
                P("base", "str", "Only pull requests targeting this branch"),
                P("limit", "int", "Maximum number to return"),
            ],
        ),
        T(
            "merge_pull_request",
            "Merge a pull request using merge, squash or rebase.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("number", "int", "Pull request number", True),
                P("method", "str", "merge, squash or rebase (default merge)"),
                P("commit_message", "str", "Commit message for the merge commit"),
            ],
            '{"merged": true, "sha": "9f2c1e7", "message": "Pull Request successfully merged"}',
        ),
        T(
            "get_file_contents",
            "Read a file or directory listing from a repository at a ref.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("path", "str", "Path inside the repository", True),
                P("ref", "str", "Branch, tag or commit SHA (default: default branch)"),
            ],
        ),
        T(
            "search_code",
            "Search code across GitHub with the code search syntax.",
            [
                P("query", "str", "Search query, e.g. 'retry_policy repo:acme/webapp'", True),
                P("repo", "str", "Restrict to one repository"),
                P("language", "str", "Restrict to a language"),
                P("limit", "int", "Maximum number of results"),
            ],
        ),
        T(
            "list_commits",
            "List commits on a branch, optionally only those touching a path.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("sha", "str", "Branch name or commit SHA to start from"),
                P("path", "str", "Only commits touching this path"),
                P("limit", "int", "Maximum number of commits"),
            ],
        ),
        T(
            "create_branch",
            "Create a new branch from another branch.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("branch", "str", "Name of the new branch", True),
                P("from_branch", "str", "Source branch (default: default branch)"),
            ],
        ),
        T(
            "list_workflow_runs",
            "List GitHub Actions workflow runs for a repository.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("workflow", "str", "Workflow file name or id, e.g. ci.yml"),
                P("status", "str", "queued, in_progress, completed, success, failure"),
                P("limit", "int", "Maximum number of runs"),
            ],
        ),
        T(
            "push_files",
            "Create or update several files in one commit on a branch.",
            [
                P("repo", "str", "Repository in owner/name form", True),
                P("branch", "str", "Branch to commit to", True),
                P("message", "str", "Commit message", True),
                P(
                    "files",
                    arr(
                        obj(
                            {
                                "path": ("str", "File path in the repository"),
                                "content": ("str", "Full new file content"),
                            },
                            ["path", "content"],
                        )
                    ),
                    "Files to write",
                    True,
                ),
            ],
            '{"commit": "b81d0aa", "files": 2, "branch": "docs-update"}',
        ),
    ],
)

SERVERS["slack"] = (
    "Slack workspace messaging: channels, threads, users and files.",
    [
        T(
            "send_message",
            "Post a message to a channel or user. Supports threads and Block Kit blocks.",
            [
                P("channel", "str", "Channel name (#releases) or id (C0123)", True),
                P("text", "str", "Message text (mrkdwn)", True),
                P("thread_ts", "str", "Reply in the thread of this message timestamp"),
                P(
                    "blocks",
                    arr(
                        obj(
                            {
                                "type": ("str", "Block type, e.g. section, divider"),
                                "text": ("str", "Block text (mrkdwn)"),
                            },
                            ["type"],
                        )
                    ),
                    "Optional Block Kit blocks",
                ),
            ],
            '{"ok": true, "channel": "C0RELEASES", "ts": "1758537600.000100"}',
        ),
        T(
            "list_channels",
            "List channels in the workspace.",
            [
                P("types", "str", "public_channel, private_channel or both, comma separated"),
                P("limit", "int", "Maximum number of channels"),
                P("cursor", "str", "Pagination cursor"),
            ],
        ),
        T(
            "get_channel_history",
            "Fetch recent messages from a channel.",
            [
                P("channel", "str", "Channel name or id", True),
                P("limit", "int", "Number of messages (default 50)"),
                P("oldest", "str", "Only messages after this timestamp"),
                P("latest", "str", "Only messages before this timestamp"),
            ],
        ),
        T(
            "reply_in_thread",
            "Reply to a message in its thread.",
            [
                P("channel", "str", "Channel name or id", True),
                P("thread_ts", "str", "Timestamp of the parent message", True),
                P("text", "str", "Reply text", True),
            ],
        ),
        T(
            "add_reaction",
            "Add an emoji reaction to a message.",
            [
                P("channel", "str", "Channel name or id", True),
                P("timestamp", "str", "Message timestamp", True),
                P("emoji", "str", "Emoji name without colons, e.g. white_check_mark", True),
            ],
        ),
        T(
            "search_messages",
            "Search messages across the workspace.",
            [
                P("query", "str", "Search query; supports in:#channel from:@user", True),
                P("channel", "str", "Restrict to this channel"),
                P("from_user", "str", "Restrict to messages from this user"),
                P("limit", "int", "Maximum number of results"),
            ],
        ),
        T(
            "list_users",
            "List workspace members.",
            [P("limit", "int", "Maximum number of users"), P("cursor", "str", "Pagination cursor")],
        ),
        T(
            "get_user_profile",
            "Get a user's profile: display name, email, title, timezone.",
            [P("user_id", "str", "User id, e.g. U0123", True)],
        ),
        T(
            "upload_file",
            "Upload a file and share it to channels.",
            [
                P("channels", "str[]", "Channels to share the file in", True),
                P("filename", "str", "File name", True),
                P("content", "str", "File content (text)", True),
                P("title", "str", "Title shown in Slack"),
            ],
        ),
        T(
            "set_channel_topic",
            "Set the topic of a channel.",
            [P("channel", "str", "Channel name or id", True), P("topic", "str", "New topic", True)],
        ),
        T(
            "create_channel",
            "Create a public or private channel and optionally invite members.",
            [
                P("name", "str", "Channel name (lowercase, no spaces)", True),
                P("is_private", "bool", "Create as private channel"),
                P("members", "str[]", "User ids to invite"),
            ],
        ),
        T(
            "schedule_message",
            "Schedule a message to be posted at a future time.",
            [
                P("channel", "str", "Channel name or id", True),
                P("text", "str", "Message text", True),
                P("post_at", "int", "Unix timestamp when to post", True),
            ],
        ),
    ],
)

SERVERS["jira"] = (
    "Jira issues, sprints, boards and worklogs.",
    [
        T(
            "create_issue",
            "Create a Jira issue in a project.",
            [
                P("project_key", "str", "Project key, e.g. PAY", True),
                P("summary", "str", "Issue summary", True),
                P("issue_type", "str", "Bug, Task, Story, Epic", True),
                P("description", "str", "Description (Jira wiki markup or plain text)"),
                P("priority", "str", "Highest, High, Medium, Low, Lowest"),
                P("labels", "str[]", "Labels to apply"),
                P("assignee", "str", "Account id or email of the assignee"),
            ],
        ),
        T(
            "get_issue",
            "Get a Jira issue with its fields, status and comments.",
            [
                P("issue_key", "str", "Issue key, e.g. PAY-731", True),
                P("fields", "str[]", "Fields to return (default all)"),
            ],
        ),
        T(
            "update_issue",
            "Update fields of an existing issue.",
            [
                P("issue_key", "str", "Issue key, e.g. PAY-731", True),
                P(
                    "fields",
                    obj(
                        {
                            "summary": ("str", "New summary"),
                            "description": ("str", "New description"),
                            "priority": ("str", "New priority name"),
                            "labels": ("str[]", "Replace labels"),
                        }
                    ),
                    "Fields to change",
                    True,
                ),
            ],
        ),
        T(
            "transition_issue",
            "Move an issue to another status (e.g. In Progress, Done).",
            [
                P("issue_key", "str", "Issue key", True),
                P("transition", "str", "Target status or transition name", True),
                P("comment", "str", "Comment to add with the transition"),
            ],
        ),
        T(
            "search_issues",
            "Search issues with JQL.",
            [
                P("jql", "str", "JQL query, e.g. project = PAY AND status != Done", True),
                P("max_results", "int", "Maximum number of issues (default 50)"),
                P("fields", "str[]", "Fields to return"),
            ],
            '{"total": 2, "issues": [{"key": "PAY-731", "summary": "Refund webhook retries", '
            '"status": "In Progress"}, {"key": "PAY-702", "summary": "3DS fallback", '
            '"status": "To Do"}]}',
        ),
        T(
            "add_comment",
            "Add a comment to an issue.",
            [
                P("issue_key", "str", "Issue key, e.g. PAY-731", True),
                P("body", "str", "Comment text", True),
            ],
            '{"id": "10457", "issue": "PAY-731", "created": "2026-09-22T08:30:00Z"}',
        ),
        T(
            "assign_issue",
            "Assign an issue to a user.",
            [
                P("issue_key", "str", "Issue key", True),
                P("assignee", "str", "Account id or email; empty to unassign", True),
            ],
        ),
        T(
            "list_sprints",
            "List sprints of a board.",
            [
                P("board_id", "int", "Board id", True),
                P("state", "str", "future, active, closed"),
            ],
        ),
        T(
            "move_to_sprint",
            "Move issues into a sprint.",
            [
                P("sprint_id", "int", "Sprint id", True),
                P("issue_keys", "str[]", "Issue keys to move", True),
            ],
        ),
        T(
            "create_subtask",
            "Create a sub-task under a parent issue.",
            [
                P("parent_key", "str", "Parent issue key", True),
                P("summary", "str", "Sub-task summary", True),
                P("description", "str", "Sub-task description"),
            ],
        ),
        T(
            "log_work",
            "Log time spent on an issue.",
            [
                P("issue_key", "str", "Issue key", True),
                P("time_spent", "str", "Duration like 2h 30m", True),
                P("comment", "str", "Worklog comment"),
                P("started", "str", "ISO-8601 start time (default now)"),
            ],
        ),
        T(
            "list_projects",
            "List Jira projects visible to the user.",
            [P("limit", "int", "Maximum number of projects")],
        ),
    ],
)

SERVERS["stripe"] = (
    "Stripe payments: customers, payment intents, refunds, invoices, subscriptions.",
    [
        T(
            "create_customer",
            "Create a Stripe customer.",
            [
                P("email", "str", "Customer email", True),
                P("name", "str", "Customer name"),
                P("description", "str", "Internal description"),
                P("metadata", "obj", "Key-value metadata"),
            ],
        ),
        T(
            "get_customer",
            "Retrieve a customer by id.",
            [P("customer_id", "str", "Customer id, e.g. cus_ABC123", True)],
        ),
        T(
            "list_customers",
            "List customers, optionally filtered by email.",
            [
                P("email", "str", "Exact email to match"),
                P("limit", "int", "Maximum number (default 10)"),
                P("starting_after", "str", "Cursor: customer id to start after"),
            ],
        ),
        T(
            "create_payment_intent",
            "Create a PaymentIntent to collect a payment.",
            [
                P("amount", "int", "Amount in the smallest currency unit (cents)", True),
                P("currency", "str", "Three-letter ISO currency, e.g. usd", True),
                P("customer", "str", "Customer id"),
                P("payment_method", "str", "Payment method id"),
                P("description", "str", "Description shown on statements"),
                P("capture_method", "str", "automatic or manual"),
            ],
        ),
        T(
            "capture_payment_intent",
            "Capture a previously authorized PaymentIntent.",
            [
                P("payment_intent_id", "str", "PaymentIntent id, e.g. pi_3Nq7...", True),
                P("amount_to_capture", "int", "Partial amount in cents (default full)"),
            ],
        ),
        T(
            "create_refund",
            "Refund a payment, fully or partially, by PaymentIntent or charge id.",
            [
                P("payment_intent", "str", "PaymentIntent id to refund"),
                P("charge", "str", "Charge id to refund (alternative to payment_intent)"),
                P("amount", "int", "Amount in cents; omit for a full refund"),
                P("reason", "str", "duplicate, fraudulent or requested_by_customer"),
            ],
            '{"id": "re_3Nq7Yz2eZvKYlo2C0aBcDeFg", "status": "succeeded", "amount": 4900, '
            '"payment_intent": "pi_3Nq7Yz2eZvKYlo2C"}',
        ),
        T(
            "list_charges",
            "List charges, optionally for one customer.",
            [
                P("customer", "str", "Customer id"),
                P("limit", "int", "Maximum number"),
                P("created_after", "int", "Unix timestamp lower bound"),
            ],
        ),
        T(
            "create_invoice",
            "Create a draft invoice for a customer with line items.",
            [
                P("customer", "str", "Customer id", True),
                P("days_until_due", "int", "Payment terms in days"),
                P("auto_advance", "bool", "Automatically finalize and send"),
                P(
                    "line_items",
                    arr(
                        obj(
                            {
                                "price": ("str", "Price id"),
                                "quantity": ("int", "Quantity"),
                            },
                            ["price"],
                        )
                    ),
                    "Invoice lines",
                ),
            ],
        ),
        T(
            "finalize_invoice",
            "Finalize a draft invoice so it can be paid.",
            [P("invoice_id", "str", "Invoice id, e.g. in_1Pq...", True)],
        ),
        T(
            "create_subscription",
            "Subscribe a customer to one or more prices.",
            [
                P("customer", "str", "Customer id", True),
                P(
                    "items",
                    arr(
                        obj(
                            {"price": ("str", "Price id"), "quantity": ("int", "Quantity")},
                            ["price"],
                        )
                    ),
                    "Subscription items",
                    True,
                ),
                P("trial_period_days", "int", "Free trial length in days"),
            ],
        ),
        T(
            "cancel_subscription",
            "Cancel a subscription immediately or at the end of the current period.",
            [
                P("subscription_id", "str", "Subscription id, e.g. sub_1PqRst", True),
                P("at_period_end", "bool", "Cancel at period end instead of now"),
            ],
            '{"id": "sub_1PqRst", "status": "active", "cancel_at_period_end": true, '
            '"current_period_end": 1761004800}',
        ),
        T(
            "list_products",
            "List products.",
            [P("active", "bool", "Only active products"), P("limit", "int", "Maximum number")],
        ),
        T(
            "create_price",
            "Create a one-time or recurring price for a product.",
            [
                P("product", "str", "Product id", True),
                P("unit_amount", "int", "Amount in cents", True),
                P("currency", "str", "ISO currency code", True),
                P(
                    "recurring",
                    obj(
                        {
                            "interval": ("str", "day, week, month or year"),
                            "interval_count": ("int", "Number of intervals between billings"),
                        },
                        ["interval"],
                    ),
                    "Recurring billing settings; omit for one-time",
                ),
            ],
        ),
    ],
)

SERVERS["postgres"] = (
    "PostgreSQL database: queries, schema inspection, indexes and diagnostics.",
    [
        T(
            "run_query",
            "Run a read-only SQL query and return rows as JSON.",
            [
                P("sql", "str", "SQL statement (SELECT only)", True),
                P("params", "str[]", "Positional parameters for $1, $2, ..."),
            ],
        ),
        T("list_schemas", "List schemas in the current database.", []),
        T(
            "list_tables",
            "List tables (and views) in a schema.",
            [P("schema", "str", "Schema name (default public)")],
            '["events", "sessions", "daily_active_users", "funnel_steps"]',
        ),
        T(
            "describe_table",
            "Show columns, types, nullability and defaults of a table.",
            [
                P("table", "str", "Table name", True),
                P("schema", "str", "Schema name (default public)"),
            ],
        ),
        T(
            "explain_query",
            "Return the execution plan of a query.",
            [
                P("sql", "str", "SQL statement", True),
                P("analyze", "bool", "Actually run it and report real timings"),
            ],
        ),
        T(
            "list_indexes",
            "List indexes on a table with their definitions and sizes.",
            [
                P("table", "str", "Table name", True),
                P("schema", "str", "Schema name (default public)"),
            ],
            '[{"name": "orders_pkey", "definition": "UNIQUE (id)", "size": "42 MB"}, '
            '{"name": "orders_customer_id_idx", "definition": "(customer_id)", "size": "18 MB"}, '
            '{"name": "orders_created_at_idx", "definition": "(created_at DESC)", '
            '"size": "15 MB"}]',
        ),
        T(
            "get_table_stats",
            "Row estimate, table and index size, last vacuum/analyze of a table.",
            [
                P("table", "str", "Table name", True),
                P("schema", "str", "Schema name (default public)"),
            ],
        ),
        T(
            "list_slow_queries",
            "List the slowest recent queries from pg_stat_statements.",
            [
                P("min_duration_ms", "int", "Only queries slower than this mean time"),
                P("limit", "int", "Maximum number of queries"),
            ],
        ),
        T(
            "list_active_connections",
            "List active backends with their current query and wait state.",
            [P("database", "str", "Restrict to one database")],
        ),
        T(
            "cancel_backend",
            "Cancel the running query of a backend process.",
            [P("pid", "int", "Backend process id", True)],
        ),
        T(
            "create_index",
            "Create an index on a table.",
            [
                P("table", "str", "Table name", True),
                P("columns", "str[]", "Columns in index order", True),
                P("name", "str", "Index name (default generated)"),
                P("unique", "bool", "Create a unique index"),
                P("concurrently", "bool", "Build without locking writes"),
            ],
        ),
        T(
            "export_query_csv",
            "Run a query and write the result to a CSV file.",
            [
                P("sql", "str", "SQL statement", True),
                P("path", "str", "Destination file path", True),
                P("delimiter", "str", "Field delimiter (default ,)"),
            ],
        ),
    ],
)

SERVERS["s3"] = (
    "Object storage (S3-compatible): buckets, objects, presigned URLs, lifecycle.",
    [
        T("list_buckets", "List all buckets in the account.", []),
        T(
            "create_bucket",
            "Create a bucket.",
            [
                P("name", "str", "Bucket name", True),
                P("region", "str", "Region, e.g. eu-west-1"),
                P("versioning", "bool", "Enable object versioning"),
            ],
        ),
        T(
            "list_objects",
            "List objects in a bucket under a prefix.",
            [
                P("bucket", "str", "Bucket name", True),
                P("prefix", "str", "Key prefix, e.g. reports/2026-09/"),
                P("max_keys", "int", "Maximum number of keys (default 1000)"),
                P("continuation_token", "str", "Pagination token"),
            ],
        ),
        T(
            "get_object",
            "Download an object's content (text) or a byte range.",
            [
                P("bucket", "str", "Bucket name", True),
                P("key", "str", "Object key", True),
                P("range", "str", "HTTP range, e.g. bytes=0-1023"),
            ],
        ),
        T(
            "put_object",
            "Upload an object.",
            [
                P("bucket", "str", "Bucket name", True),
                P("key", "str", "Object key", True),
                P("body", "str", "Content to upload", True),
                P("content_type", "str", "MIME type"),
                P("metadata", "obj", "User metadata key-values"),
            ],
        ),
        T(
            "delete_object",
            "Delete an object or one version of it.",
            [
                P("bucket", "str", "Bucket name", True),
                P("key", "str", "Object key", True),
                P("version_id", "str", "Version to delete"),
            ],
        ),
        T(
            "copy_object",
            "Copy an object within or across buckets.",
            [
                P("source_bucket", "str", "Source bucket", True),
                P("source_key", "str", "Source key", True),
                P("dest_bucket", "str", "Destination bucket", True),
                P("dest_key", "str", "Destination key", True),
            ],
        ),
        T(
            "generate_presigned_url",
            "Create a time-limited URL to download or upload an object without credentials.",
            [
                P("bucket", "str", "Bucket name", True),
                P("key", "str", "Object key", True),
                P("expires_in", "int", "Lifetime in seconds (default 3600)"),
                P("method", "str", "GET or PUT (default GET)"),
            ],
            '{"url": "https://acme-reports.s3.amazonaws.com/reports/2026-09/summary.pdf?'
            'X-Amz-Expires=3600&X-Amz-Signature=abc123", "expires_in": 3600}',
        ),
        T(
            "get_object_metadata",
            "HEAD an object: size, content type, ETag, last modified, user metadata.",
            [P("bucket", "str", "Bucket name", True), P("key", "str", "Object key", True)],
        ),
        T(
            "set_bucket_lifecycle",
            "Replace the lifecycle rules of a bucket.",
            [
                P("bucket", "str", "Bucket name", True),
                P(
                    "rules",
                    arr(
                        obj(
                            {
                                "prefix": ("str", "Key prefix the rule applies to"),
                                "expiration_days": ("int", "Delete objects after N days"),
                                "transition_storage_class": (
                                    "str",
                                    "Storage class to move to, e.g. GLACIER",
                                ),
                            },
                            ["prefix"],
                        )
                    ),
                    "Lifecycle rules",
                    True,
                ),
            ],
        ),
        T(
            "list_object_versions",
            "List versions of objects under a prefix in a versioned bucket.",
            [
                P("bucket", "str", "Bucket name", True),
                P("prefix", "str", "Key prefix"),
                P("max_keys", "int", "Maximum number of versions"),
            ],
        ),
        T(
            "sync_prefix",
            "Copy everything under a prefix to another bucket/prefix, optionally deleting extras.",
            [
                P("source_bucket", "str", "Source bucket", True),
                P("source_prefix", "str", "Source prefix", True),
                P("dest_bucket", "str", "Destination bucket", True),
                P("dest_prefix", "str", "Destination prefix", True),
                P("delete", "bool", "Delete destination objects missing from source"),
            ],
        ),
    ],
)

SERVERS["sentry"] = (
    "Sentry error monitoring: issues, events, releases.",
    [
        T(
            "list_issues",
            "List issues in a project with a search query.",
            [
                P("project", "str", "Project slug", True),
                P("query", "str", "Sentry search, e.g. is:unresolved level:error"),
                P("status", "str", "unresolved, resolved, ignored"),
                P("sort", "str", "date, new, freq, priority"),
                P("limit", "int", "Maximum number of issues"),
            ],
        ),
        T(
            "get_issue",
            "Get an issue: title, culprit, counts, first/last seen, status.",
            [P("issue_id", "str", "Issue id", True)],
        ),
        T(
            "list_issue_events",
            "List recent events of an issue.",
            [P("issue_id", "str", "Issue id", True), P("limit", "int", "Maximum events")],
        ),
        T(
            "get_event",
            "Get one event with stack trace, tags and breadcrumbs.",
            [P("project", "str", "Project slug", True), P("event_id", "str", "Event id", True)],
        ),
        T(
            "resolve_issue",
            "Mark an issue as resolved, optionally in a specific release.",
            [
                P("issue_id", "str", "Issue id", True),
                P("in_release", "str", "Resolve in this release version, or 'latest'"),
            ],
            '{"id": "4821337", "status": "resolved", "statusDetails": {}}',
        ),
        T(
            "ignore_issue",
            "Ignore an issue for a duration.",
            [
                P("issue_id", "str", "Issue id", True),
                P("duration_minutes", "int", "Ignore for this many minutes (default forever)"),
            ],
        ),
        T(
            "assign_issue",
            "Assign an issue to a user or team.",
            [
                P("issue_id", "str", "Issue id", True),
                P("assignee", "str", "user:email or team:slug", True),
            ],
        ),
        T(
            "list_projects",
            "List projects in an organization.",
            [P("organization", "str", "Organization slug")],
        ),
        T(
            "list_releases",
            "List releases of a project.",
            [P("project", "str", "Project slug", True), P("limit", "int", "Maximum releases")],
        ),
        T(
            "create_release",
            "Create a release (version) for a project.",
            [
                P("project", "str", "Project slug", True),
                P("version", "str", "Release version string", True),
                P("ref", "str", "Commit SHA of the release"),
                P("url", "str", "Link to the release"),
            ],
        ),
        T(
            "get_issue_tags",
            "Get tag value distribution for an issue (browser, os, release, ...).",
            [P("issue_id", "str", "Issue id", True), P("key", "str", "Only this tag key")],
        ),
        T(
            "search_events",
            "Search events in a project with the Discover query syntax.",
            [
                P("project", "str", "Project slug", True),
                P("query", "str", "Discover query", True),
                P("fields", "str[]", "Fields to return"),
                P("limit", "int", "Maximum events"),
            ],
        ),
    ],
)

SERVERS["datadog"] = (
    "Datadog observability: metrics, monitors, logs, dashboards, incidents, events.",
    [
        T(
            "query_metrics",
            "Query timeseries metrics over a time range.",
            [
                P(
                    "query",
                    "str",
                    "Metric query, e.g. p95:trace.http.request.duration{service:checkout}",
                    True,
                ),
                P("from_ts", "int", "Start as Unix timestamp", True),
                P("to_ts", "int", "End as Unix timestamp", True),
            ],
            '{"series": [{"metric": "trace.http.request.duration", "scope": "service:checkout", '
            '"pointlist": [[1758531600, 0.412], [1758533400, 0.395], [1758535200, 0.428]]}]}',
        ),
        T(
            "list_metrics",
            "List metric names, optionally filtered.",
            [P("filter", "str", "Substring filter"), P("host", "str", "Only metrics from a host")],
        ),
        T(
            "submit_metric",
            "Submit custom metric points.",
            [
                P("metric", "str", "Metric name", True),
                P(
                    "points",
                    arr(
                        obj(
                            {
                                "timestamp": ("int", "Unix timestamp"),
                                "value": ("num", "Metric value"),
                            },
                            ["timestamp", "value"],
                        )
                    ),
                    "Points to submit",
                    True,
                ),
                P("type", "str", "gauge, count or rate"),
                P("tags", "str[]", "Tags like env:prod"),
            ],
        ),
        T(
            "list_monitors",
            "List monitors by name or tags.",
            [
                P("name", "str", "Substring of the monitor name"),
                P("tags", "str[]", "Scope tags"),
                P("monitor_tags", "str[]", "Monitor tags"),
            ],
        ),
        T(
            "get_monitor",
            "Get a monitor's definition and state.",
            [P("monitor_id", "int", "Monitor id", True)],
        ),
        T(
            "create_monitor",
            "Create a monitor.",
            [
                P("name", "str", "Monitor name", True),
                P("type", "str", "metric alert, log alert, service check, ...", True),
                P("query", "str", "Monitor query", True),
                P("message", "str", "Notification message with @-handles"),
                P("tags", "str[]", "Monitor tags"),
                P(
                    "thresholds",
                    obj(
                        {
                            "critical": ("num", "Critical threshold"),
                            "warning": ("num", "Warning threshold"),
                        },
                        ["critical"],
                    ),
                    "Alert thresholds",
                ),
            ],
        ),
        T(
            "mute_monitor",
            "Mute a monitor until a time, optionally for one scope.",
            [
                P("monitor_id", "int", "Monitor id", True),
                P("end", "int", "Unix timestamp when the mute ends (default indefinite)"),
                P("scope", "str", "Scope to mute, e.g. host:web-1"),
            ],
            '{"id": 998877, "overall_state": "OK", "options": {"silenced": {"*": 1758544800}}}',
        ),
        T(
            "search_logs",
            "Search logs with the Datadog log query syntax.",
            [
                P("query", "str", "Log query, e.g. service:checkout status:error", True),
                P("from_time", "str", "Start, e.g. now-15m or ISO-8601"),
                P("to_time", "str", "End, e.g. now"),
                P("limit", "int", "Maximum log lines"),
                P("sort", "str", "timestamp or -timestamp"),
            ],
        ),
        T(
            "list_dashboards",
            "List dashboards.",
            [P("filter_shared", "bool", "Only shared dashboards")],
        ),
        T(
            "get_dashboard",
            "Get a dashboard definition.",
            [P("dashboard_id", "str", "Dashboard id", True)],
        ),
        T(
            "list_incidents",
            "List incidents.",
            [P("state", "str", "active, stable, resolved"), P("limit", "int", "Maximum incidents")],
        ),
        T(
            "create_event",
            "Post an event to the event stream.",
            [
                P("title", "str", "Event title", True),
                P("text", "str", "Event body", True),
                P("alert_type", "str", "error, warning, info, success"),
                P("tags", "str[]", "Tags"),
                P("priority", "str", "normal or low"),
            ],
        ),
    ],
)

SERVERS["notion"] = (
    "Notion pages, databases, blocks and comments.",
    [
        T(
            "search",
            "Search pages and databases by title.",
            [
                P("query", "str", "Search text", True),
                P("filter_type", "str", "page or database"),
                P("page_size", "int", "Maximum results"),
            ],
        ),
        T(
            "get_page",
            "Get a page's properties and metadata.",
            [P("page_id", "str", "Page id", True)],
        ),
        T(
            "create_page",
            "Create a page under a parent page or database.",
            [
                P("parent_id", "str", "Parent page or database id", True),
                P("title", "str", "Page title", True),
                P("content_markdown", "str", "Initial body as Markdown"),
                P("properties", "obj", "Database properties when the parent is a database"),
            ],
            '{"id": "c41d9e2a-7f3b-4b2e-9a1d-0e5f6a7b8c9d", "url": "https://notion.so/Q4-Roadmap-c41d9e2a"}',
        ),
        T(
            "update_page_properties",
            "Update properties of a page.",
            [
                P("page_id", "str", "Page id", True),
                P("properties", "obj", "Property values to set", True),
            ],
        ),
        T(
            "append_blocks",
            "Append content blocks to a page.",
            [
                P("page_id", "str", "Page id", True),
                P(
                    "blocks",
                    arr(
                        obj(
                            {
                                "type": ("str", "paragraph, heading_2, bulleted_list_item, ..."),
                                "text": ("str", "Block text"),
                            },
                            ["type", "text"],
                        )
                    ),
                    "Blocks to append",
                    True,
                ),
            ],
        ),
        T(
            "get_block_children",
            "List child blocks of a page or block.",
            [
                P("block_id", "str", "Block or page id", True),
                P("page_size", "int", "Maximum blocks"),
            ],
        ),
        T(
            "query_database",
            "Query a database with filters and sorts.",
            [
                P("database_id", "str", "Database id", True),
                P("filter", "obj", "Notion filter object"),
                P(
                    "sorts",
                    arr(
                        obj(
                            {
                                "property": ("str", "Property to sort by"),
                                "direction": ("str", "ascending or descending"),
                            },
                            ["property"],
                        )
                    ),
                    "Sort order",
                ),
                P("page_size", "int", "Maximum rows"),
            ],
        ),
        T(
            "create_database_item",
            "Add a row to a database.",
            [
                P("database_id", "str", "Database id", True),
                P("properties", "obj", "Property values", True),
            ],
        ),
        T(
            "get_database",
            "Get a database's schema.",
            [P("database_id", "str", "Database id", True)],
        ),
        T("archive_page", "Archive (soft delete) a page.", [P("page_id", "str", "Page id", True)]),
        T("list_users", "List workspace users.", [P("page_size", "int", "Maximum users")]),
        T(
            "add_comment",
            "Add a comment to a page.",
            [P("page_id", "str", "Page id", True), P("text", "str", "Comment text", True)],
        ),
    ],
)

SERVERS["linear"] = (
    "Linear issue tracking: issues, teams, projects, cycles, labels.",
    [
        T(
            "create_issue",
            "Create an issue in a team.",
            [
                P("team_id", "str", "Team id or key, e.g. ENG", True),
                P("title", "str", "Issue title", True),
                P("description", "str", "Markdown description"),
                P("priority", "int", "0 none, 1 urgent, 2 high, 3 medium, 4 low"),
                P("assignee_id", "str", "User id to assign"),
                P("label_ids", "str[]", "Label ids"),
                P("project_id", "str", "Project id"),
            ],
            '{"id": "9d1f...", "identifier": "ENG-1187", "url": "https://linear.app/acme/issue/ENG-1187"}',
        ),
        T(
            "update_issue",
            "Update an issue's fields.",
            [
                P("issue_id", "str", "Issue id or identifier", True),
                P("title", "str", "New title"),
                P("description", "str", "New description"),
                P("state_id", "str", "Workflow state id"),
                P("priority", "int", "New priority"),
                P("assignee_id", "str", "New assignee"),
            ],
        ),
        T(
            "get_issue",
            "Get an issue by id or identifier.",
            [P("issue_id", "str", "Issue id or ENG-123", True)],
        ),
        T(
            "search_issues",
            "Search issues by text with optional filters.",
            [
                P("query", "str", "Search text", True),
                P("team_id", "str", "Restrict to a team"),
                P("state", "str", "Workflow state name"),
                P("assignee_id", "str", "Restrict to an assignee"),
                P("limit", "int", "Maximum results"),
            ],
        ),
        T("list_teams", "List teams in the workspace.", []),
        T(
            "list_workflow_states",
            "List workflow states of a team.",
            [P("team_id", "str", "Team id", True)],
        ),
        T(
            "add_comment",
            "Comment on an issue.",
            [P("issue_id", "str", "Issue id", True), P("body", "str", "Markdown body", True)],
        ),
        T(
            "list_projects",
            "List projects, optionally by team or state.",
            [P("team_id", "str", "Team id"), P("state", "str", "planned, started, completed")],
        ),
        T(
            "create_project",
            "Create a project.",
            [
                P("name", "str", "Project name", True),
                P("team_ids", "str[]", "Teams the project belongs to", True),
                P("description", "str", "Description"),
                P("target_date", "str", "Target date YYYY-MM-DD"),
            ],
        ),
        T(
            "list_cycles",
            "List cycles (sprints) of a team.",
            [
                P("team_id", "str", "Team id", True),
                P("active_only", "bool", "Only the active cycle"),
            ],
        ),
        T(
            "move_issue_to_cycle",
            "Put an issue into a cycle.",
            [P("issue_id", "str", "Issue id", True), P("cycle_id", "str", "Cycle id", True)],
        ),
        T("list_labels", "List issue labels.", [P("team_id", "str", "Restrict to a team")]),
    ],
)

SERVERS["calendar"] = (
    "Calendar: events, availability, invitations.",
    [
        T("list_calendars", "List calendars the user can access.", []),
        T(
            "list_events",
            "List events in a time window.",
            [
                P("calendar_id", "str", "Calendar id (default primary)"),
                P("time_min", "str", "ISO-8601 lower bound"),
                P("time_max", "str", "ISO-8601 upper bound"),
                P("query", "str", "Free-text filter"),
                P("max_results", "int", "Maximum events"),
            ],
        ),
        T(
            "get_event",
            "Get one event.",
            [P("calendar_id", "str", "Calendar id", True), P("event_id", "str", "Event id", True)],
        ),
        T(
            "create_event",
            "Create an event with attendees and reminders.",
            [
                P("calendar_id", "str", "Calendar id (primary for the user's own)", True),
                P("summary", "str", "Event title", True),
                P("start", "str", "Start as ISO-8601 datetime", True),
                P("end", "str", "End as ISO-8601 datetime", True),
                P("description", "str", "Event description"),
                P("location", "str", "Location or meeting link"),
                P(
                    "attendees",
                    arr(
                        obj(
                            {
                                "email": ("str", "Attendee email"),
                                "optional": ("bool", "Optional attendee"),
                            },
                            ["email"],
                        )
                    ),
                    "Attendees to invite",
                ),
                P("reminders_minutes", "int[]", "Popup reminders, minutes before start"),
            ],
            '{"id": "evt_7a1c", "htmlLink": "https://calendar.example/event?eid=evt_7a1c", '
            '"status": "confirmed"}',
        ),
        T(
            "update_event",
            "Change an event's fields.",
            [
                P("calendar_id", "str", "Calendar id", True),
                P("event_id", "str", "Event id", True),
                P("summary", "str", "New title"),
                P("start", "str", "New start"),
                P("end", "str", "New end"),
                P("description", "str", "New description"),
                P("location", "str", "New location"),
            ],
        ),
        T(
            "delete_event",
            "Delete an event.",
            [
                P("calendar_id", "str", "Calendar id", True),
                P("event_id", "str", "Event id", True),
                P("send_updates", "str", "all, externalOnly or none"),
            ],
        ),
        T(
            "find_free_slots",
            "Find times when all attendees are free.",
            [
                P("attendees", "str[]", "Attendee emails", True),
                P("duration_minutes", "int", "Meeting length", True),
                P("time_min", "str", "Search window start", True),
                P("time_max", "str", "Search window end", True),
                P("working_hours_only", "bool", "Restrict to 09:00-18:00 local"),
            ],
        ),
        T(
            "respond_to_invite",
            "Accept, decline or tentatively accept an invitation.",
            [
                P("calendar_id", "str", "Calendar id", True),
                P("event_id", "str", "Event id", True),
                P("response", "str", "accepted, declined or tentative", True),
            ],
        ),
        T(
            "create_recurring_event",
            "Create a repeating event from an RRULE.",
            [
                P("calendar_id", "str", "Calendar id", True),
                P("summary", "str", "Event title", True),
                P("start", "str", "First occurrence start", True),
                P("end", "str", "First occurrence end", True),
                P("rrule", "str", "RRULE string, e.g. FREQ=WEEKLY;BYDAY=MO", True),
            ],
        ),
        T(
            "list_upcoming",
            "List events in the next N days.",
            [P("calendar_id", "str", "Calendar id"), P("days", "int", "Days ahead (default 7)")],
        ),
        T(
            "add_attendees",
            "Invite more people to an existing event.",
            [
                P("calendar_id", "str", "Calendar id", True),
                P("event_id", "str", "Event id", True),
                P("emails", "str[]", "Emails to add", True),
            ],
        ),
        T(
            "set_out_of_office",
            "Create an out-of-office block that auto-declines invitations.",
            [
                P("start", "str", "Start datetime", True),
                P("end", "str", "End datetime", True),
                P("message", "str", "Decline message"),
            ],
        ),
    ],
)

SERVERS["email"] = (
    "Mailbox (IMAP/SMTP): send, search, reply, organize messages.",
    [
        T(
            "send_email",
            "Send an email.",
            [
                P("to", "str[]", "Recipient addresses", True),
                P("subject", "str", "Subject line", True),
                P("body", "str", "Body (plain text or HTML)", True),
                P("cc", "str[]", "CC addresses"),
                P("bcc", "str[]", "BCC addresses"),
                P("html", "bool", "Body is HTML"),
                P(
                    "attachments",
                    arr(
                        obj(
                            {
                                "filename": ("str", "Attachment file name"),
                                "content_base64": ("str", "Base64-encoded content"),
                            },
                            ["filename", "content_base64"],
                        )
                    ),
                    "Attachments",
                ),
            ],
            '{"message_id": "<20260922083000.1042@acme.io>", "accepted": ["billing@acme.io"]}',
        ),
        T(
            "list_messages",
            "List messages in a folder.",
            [
                P("folder", "str", "Folder name (default INBOX)"),
                P("query", "str", "Text filter"),
                P("unread_only", "bool", "Only unread"),
                P("limit", "int", "Maximum messages"),
            ],
        ),
        T(
            "get_message",
            "Read one message with headers and body.",
            [
                P("message_id", "str", "Message id", True),
                P("include_attachments", "bool", "Include attachment metadata"),
            ],
        ),
        T(
            "reply_to_message",
            "Reply to a message, keeping the thread.",
            [
                P("message_id", "str", "Message to reply to", True),
                P("body", "str", "Reply body", True),
                P("reply_all", "bool", "Reply to all recipients"),
            ],
        ),
        T(
            "forward_message",
            "Forward a message.",
            [
                P("message_id", "str", "Message to forward", True),
                P("to", "str[]", "Recipients", True),
                P("note", "str", "Text to add above the forwarded message"),
            ],
        ),
        T(
            "search_messages",
            "Search messages across folders.",
            [
                P("query", "str", "Search text", True),
                P("folder", "str", "Restrict to a folder"),
                P("from_addr", "str", "Sender filter"),
                P("since", "str", "Only messages after this date"),
                P("limit", "int", "Maximum results"),
            ],
        ),
        T(
            "move_message",
            "Move a message to another folder.",
            [P("message_id", "str", "Message id", True), P("folder", "str", "Target folder", True)],
        ),
        T(
            "mark_as_read",
            "Mark messages read or unread.",
            [
                P("message_ids", "str[]", "Message ids", True),
                P("read", "bool", "true = read, false = unread (default true)"),
            ],
        ),
        T("list_folders", "List mailbox folders.", []),
        T(
            "create_draft",
            "Save a draft without sending.",
            [
                P("to", "str[]", "Recipients", True),
                P("subject", "str", "Subject", True),
                P("body", "str", "Body", True),
            ],
        ),
        T(
            "delete_message",
            "Move a message to Trash or delete permanently.",
            [P("message_id", "str", "Message id", True), P("permanent", "bool", "Skip Trash")],
        ),
        T(
            "download_attachment",
            "Save an attachment to disk.",
            [
                P("message_id", "str", "Message id", True),
                P("attachment_id", "str", "Attachment id", True),
                P("path", "str", "Destination path", True),
            ],
        ),
    ],
)

SERVERS["websearch"] = (
    "Web search and page fetching, plus small utilities (translate, currency, definitions).",
    [
        T(
            "web_search",
            "Search the web and return titles, URLs and snippets.",
            [
                P("query", "str", "Search query", True),
                P("num_results", "int", "Number of results (default 5)"),
                P("site", "str", "Restrict to a domain"),
                P("recency", "str", "day, week, month or year"),
            ],
            '[{"title": "PostgreSQL 17 Released!", "url": "https://www.postgresql.org/about/news/'
            'postgresql-17-released-2936/", "snippet": "PostgreSQL 17 includes ..."}]',
        ),
        T(
            "fetch_page",
            "Fetch a URL and return its readable text.",
            [
                P("url", "str", "Page URL", True),
                P("max_chars", "int", "Truncate to this many characters"),
                P("as_markdown", "bool", "Convert to Markdown"),
            ],
        ),
        T(
            "news_search",
            "Search recent news articles.",
            [
                P("query", "str", "Search query", True),
                P("days", "int", "Only the last N days"),
                P("num_results", "int", "Number of results"),
            ],
        ),
        T(
            "image_search",
            "Search images.",
            [
                P("query", "str", "Search query", True),
                P("num_results", "int", "Number of results"),
                P("size", "str", "small, medium, large"),
            ],
        ),
        T(
            "extract_links",
            "List links found on a page.",
            [
                P("url", "str", "Page URL", True),
                P("same_domain_only", "bool", "Skip external links"),
            ],
        ),
        T(
            "summarize_url",
            "Fetch a page and summarize it.",
            [P("url", "str", "Page URL", True), P("max_words", "int", "Summary length")],
        ),
        T(
            "check_url_status",
            "HTTP status, redirects and response time of a URL.",
            [P("url", "str", "URL", True)],
        ),
        T(
            "search_academic",
            "Search academic papers.",
            [
                P("query", "str", "Search query", True),
                P("year_from", "int", "Only papers from this year on"),
                P("num_results", "int", "Number of results"),
            ],
        ),
        T(
            "get_sitemap",
            "Fetch and parse a site's sitemap.xml.",
            [P("url", "str", "Site root URL", True)],
        ),
        T(
            "translate_text",
            "Translate text between languages.",
            [
                P("text", "str", "Text to translate", True),
                P("target_lang", "str", "Target language code, e.g. ko", True),
                P("source_lang", "str", "Source language code (auto-detect if omitted)"),
            ],
        ),
        T(
            "define_word",
            "Dictionary definition of a word.",
            [P("word", "str", "Word to define", True), P("language", "str", "Language code")],
        ),
        T(
            "currency_convert",
            "Convert an amount between currencies at the current rate.",
            [
                P("amount", "num", "Amount to convert", True),
                P("from_currency", "str", "Source ISO code", True),
                P("to_currency", "str", "Target ISO code", True),
            ],
        ),
    ],
)

SERVERS["confluence"] = (
    "Confluence wiki: spaces, pages, labels, comments.",
    [
        T(
            "search_pages",
            "Search pages with CQL.",
            [
                P("cql", "str", 'CQL query, e.g. space = OPS AND text ~ "on-call"', True),
                P("limit", "int", "Maximum results"),
                P("expand", "str[]", "Fields to expand, e.g. body.storage"),
            ],
            '{"results": [{"id": "88211", "title": "On-call rotation and escalation", '
            '"space": "OPS"}, {"id": "88340", "title": "PagerDuty schedule how-to", '
            '"space": "OPS"}]}',
        ),
        T(
            "get_page",
            "Get a page with its body.",
            [
                P("page_id", "str", "Page id", True),
                P("body_format", "str", "storage, view or markdown"),
            ],
        ),
        T(
            "create_page",
            "Create a page in a space.",
            [
                P("space_key", "str", "Space key, e.g. OPS", True),
                P("title", "str", "Page title", True),
                P("body", "str", "Body in storage format or Markdown", True),
                P("parent_id", "str", "Parent page id"),
                P("labels", "str[]", "Labels to add"),
            ],
        ),
        T(
            "update_page",
            "Update a page's title or body (creates a new version).",
            [
                P("page_id", "str", "Page id", True),
                P("title", "str", "New title"),
                P("body", "str", "New body", True),
                P("version_comment", "str", "Change note"),
            ],
        ),
        T("delete_page", "Move a page to trash.", [P("page_id", "str", "Page id", True)]),
        T(
            "list_spaces",
            "List spaces.",
            [P("type", "str", "global or personal"), P("limit", "int", "Maximum spaces")],
        ),
        T(
            "get_page_children",
            "List child pages.",
            [P("page_id", "str", "Page id", True), P("limit", "int", "Maximum children")],
        ),
        T(
            "add_label",
            "Add labels to a page.",
            [P("page_id", "str", "Page id", True), P("labels", "str[]", "Labels", True)],
        ),
        T(
            "get_page_history",
            "List versions of a page with authors and dates.",
            [P("page_id", "str", "Page id", True), P("limit", "int", "Maximum versions")],
        ),
        T(
            "add_comment",
            "Add a comment to a page.",
            [P("page_id", "str", "Page id", True), P("body", "str", "Comment body", True)],
        ),
        T(
            "export_page_pdf",
            "Export a page as PDF to a path.",
            [P("page_id", "str", "Page id", True), P("path", "str", "Destination file", True)],
        ),
        T(
            "move_page",
            "Move a page under another parent.",
            [
                P("page_id", "str", "Page id", True),
                P("target_parent_id", "str", "New parent page id", True),
                P("position", "str", "append, above or below"),
            ],
        ),
    ],
)

SERVERS["ci"] = (
    "CI/CD: pipelines, jobs, logs, artifacts, environments, variables.",
    [
        T(
            "list_pipelines",
            "List pipelines of a project.",
            [
                P("project", "str", "Project path, e.g. acme/webapp", True),
                P("branch", "str", "Only this branch"),
                P("status", "str", "running, success, failed, canceled"),
                P("limit", "int", "Maximum pipelines"),
            ],
        ),
        T(
            "get_pipeline",
            "Get a pipeline's status, duration and stages.",
            [
                P("project", "str", "Project path", True),
                P("pipeline_id", "int", "Pipeline id", True),
            ],
        ),
        T(
            "trigger_pipeline",
            "Start a new pipeline on a branch or tag.",
            [
                P("project", "str", "Project path, e.g. acme/webapp", True),
                P("ref", "str", "Branch or tag to run on", True),
                P("variables", "obj", "Pipeline variables as key-values"),
            ],
            '{"id": 77231, "status": "pending", "ref": "release/2.4", '
            '"web_url": "https://ci.acme.io/acme/webapp/-/pipelines/77231"}',
        ),
        T(
            "cancel_pipeline",
            "Cancel a running pipeline.",
            [
                P("project", "str", "Project path", True),
                P("pipeline_id", "int", "Pipeline id", True),
            ],
        ),
        T(
            "retry_pipeline",
            "Retry failed jobs of a pipeline.",
            [
                P("project", "str", "Project path", True),
                P("pipeline_id", "int", "Pipeline id", True),
            ],
        ),
        T(
            "list_jobs",
            "List jobs of a pipeline.",
            [
                P("project", "str", "Project path", True),
                P("pipeline_id", "int", "Pipeline id", True),
                P("status", "str", "Filter by status"),
            ],
        ),
        T(
            "get_job_log",
            "Fetch the console log of a job.",
            [
                P("project", "str", "Project path", True),
                P("job_id", "int", "Job id", True),
                P("tail_lines", "int", "Only the last N lines"),
            ],
            "... 197 lines omitted ...\nFAILED tests/test_checkout.py::test_apply_coupon - "
            "AssertionError: expected 9.90, got 9.9\n1 failed, 212 passed in 41.2s\n"
            "ERROR: Job failed: exit code 1",
        ),
        T(
            "list_artifacts",
            "List artifacts produced by a job.",
            [P("project", "str", "Project path", True), P("job_id", "int", "Job id", True)],
        ),
        T(
            "download_artifact",
            "Download one artifact file from a job.",
            [
                P("project", "str", "Project path", True),
                P("job_id", "int", "Job id", True),
                P("path", "str", "Artifact path inside the job", True),
                P("dest", "str", "Local destination path", True),
            ],
        ),
        T(
            "list_environments",
            "List deployment environments.",
            [P("project", "str", "Project path", True)],
        ),
        T(
            "deploy_environment",
            "Deploy a ref to an environment.",
            [
                P("project", "str", "Project path", True),
                P("environment", "str", "Environment name, e.g. staging", True),
                P("ref", "str", "Branch, tag or SHA", True),
                P("variables", "obj", "Deployment variables"),
            ],
        ),
        T(
            "list_schedules",
            "List pipeline schedules.",
            [P("project", "str", "Project path", True)],
        ),
        T(
            "set_variable",
            "Create or update a CI variable.",
            [
                P("project", "str", "Project path", True),
                P("key", "str", "Variable name", True),
                P("value", "str", "Variable value", True),
                P("protected", "bool", "Only on protected branches"),
                P("masked", "bool", "Mask in logs"),
            ],
        ),
    ],
)


# --------------------------------------------------------------------------- build


def build() -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    servers: dict[str, str] = {}
    for server, (blurb, specs) in SERVERS.items():
        servers[server] = blurb
        for spec in specs:
            props = {p["name"]: p["schema"] for p in spec["params"]}
            required = [p["name"] for p in spec["params"] if p["required"]]
            schema: dict[str, Any] = {"type": "object", "properties": props}
            if required:
                schema["required"] = required
            tool = {
                "server": server,
                "name": spec["name"],
                "description": spec["description"],
                "inputSchema": schema,
            }
            if spec.get("result"):
                tool["result"] = spec["result"]
            tools.append(tool)
    names = [f"{t['server']}__{t['name']}" for t in tools]
    assert len(names) == len(set(names)), "duplicate tool names"
    return {
        "version": 1,
        "generated_by": "scripts/gen_synthetic_catalog.py",
        "servers": servers,
        "tools": tools,
    }


def main(argv: list[str]) -> int:
    data = build()
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT} is stale; run scripts/gen_synthetic_catalog.py")
            return 1
        print(f"{OUT} is up to date ({len(data['tools'])} tools)")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    nested = sum(
        1
        for t in data["tools"]
        for p in t["inputSchema"]["properties"].values()
        if p.get("type") == "array"
        and (p.get("items") or {}).get("type") == "object"
        or p.get("type") == "object"
        and p.get("properties")
    )
    print(
        f"wrote {OUT.relative_to(OUT.parents[2])}: {len(data['tools'])} tools, "
        f"{len(data['servers'])} servers, {nested} nested parameters"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
