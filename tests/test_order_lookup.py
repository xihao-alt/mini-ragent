"""Order lookup uses fixed SQL and the authenticated session user."""

from __future__ import annotations

from session import SessionStore
from tools import OrderLookupTool, ToolRegistry
from tools.execution_context import ExecutionContext, reset_execution_context, set_execution_context


def test_seeded_orders_and_schema(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    with store._connect() as connection:
        rows = connection.execute("SELECT order_id, user_id FROM orders ORDER BY order_id").fetchall()
    assert len(rows) == 9
    assert rows[0]["order_id"] == "NS001"
    registry = ToolRegistry()
    registry.register(OrderLookupTool(store))
    schema = registry.schemas()[0]["parameters"]
    assert schema["required"] == ["order_id"]
    assert list(schema["properties"]) == ["order_id"]


def test_lookup_scopes_to_session_user(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    tool = OrderLookupTool(store)
    token = set_execution_context(ExecutionContext("user_001", "session-a"))
    try:
        own = tool.execute(order_id="NS001")
        other = tool.execute(order_id="NS008")
        injection = tool.execute(order_id="NS001' OR 1=1 --")
    finally:
        reset_execution_context(token)
    assert own["shipping_status"] == "delayed"
    assert own["completed_charge_count"] == 2
    assert other == {"error": "Order not found for the current user"}
    assert injection == {"error": "Order not found for the current user"}
