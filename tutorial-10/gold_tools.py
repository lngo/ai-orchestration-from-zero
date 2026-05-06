"""
The gold toolkit. Plain Python functions, a JSON-schema tool list,
and a dispatcher that maps tool names to functions.
"""

import os
import requests


# -----------------------------------------------------------
# THE GOLD FUNCTIONS (same logic as Phase 1 and T9)
# -----------------------------------------------------------

def get_gold_price() -> str:
    """Fetch the current gold price in USD per troy ounce."""
    api_key = os.environ.get("GOLDAPI_KEY")
    if not api_key:
        return "Error: GOLDAPI_KEY environment variable is not set."

    url = "https://www.goldapi.io/api/XAU/USD"
    headers = {"x-access-token": api_key}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return (
            f"Gold price: ${data['price']:.2f} per troy ounce. "
            f"Previous close: ${data['prev_close_price']:.2f}. "
            f"Change: ${data['ch']:+.2f} ({data['chp']:+.2f}%). "
            f"Day range: ${data['low_price']:.2f} - ${data['high_price']:.2f}."
        )
    except Exception as e:
        return f"Error fetching gold price: {e}"


def convert_currency(amount: float, to_currency: str) -> str:
    """Convert USD to another currency using live rates."""
    url = f"https://api.frankfurter.dev/v1/latest?base=USD&symbols={to_currency.upper()}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        rate = data["rates"][to_currency.upper()]
        converted = amount * rate
        return f"USD {amount:,.2f} = {to_currency.upper()} {converted:,.2f}"
    except Exception as e:
        return f"Error: {e}"


def calculate_portfolio(ounces: float, price_per_ounce: float) -> str:
    """Calculate total value of a gold holding."""
    total = ounces * price_per_ounce
    return f"Portfolio: {ounces} oz x ${price_per_ounce:.2f} = ${total:,.2f} USD"


# -----------------------------------------------------------
# TOOL SCHEMA (Claude's expected JSON format)
# -----------------------------------------------------------

GOLD_TOOLS = [
    {
        "name": "get_gold_price",
        "description": "Fetch the current gold price in USD per troy ounce, including today's change and day range.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "convert_currency",
        "description": "Convert a USD amount to another currency using live exchange rates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount in USD"},
                "to_currency": {"type": "string", "description": "Target currency code (EUR, GBP, AUD, JPY)"},
            },
            "required": ["amount", "to_currency"],
        },
    },
    {
        "name": "calculate_portfolio",
        "description": "Calculate the total USD value of a gold holding given the number of ounces and price per ounce.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ounces": {"type": "number"},
                "price_per_ounce": {"type": "number"},
            },
            "required": ["ounces", "price_per_ounce"],
        },
    },
]


# -----------------------------------------------------------
# DISPATCHER
# -----------------------------------------------------------

def execute_gold_tool(name: str, params: dict) -> str:
    """Dispatch a tool call from Claude to the matching Python function."""
    if name == "get_gold_price":
        return get_gold_price()
    if name == "convert_currency":
        return convert_currency(**params)
    if name == "calculate_portfolio":
        return calculate_portfolio(**params)
    return f"Unknown tool: {name}"