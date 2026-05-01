"""
Gold Tools MCP Server
Exposes the Phase 1 gold tools (get_gold_price, convert_currency,
calculate_portfolio) over MCP, so they can be used by any
MCP-compatible client — including Claude Code.
"""

import os
import requests
from fastmcp import FastMCP

mcp = FastMCP(name="gold-tools")


@mcp.tool
def get_gold_price() -> str:
    """Fetch the current gold price in USD per troy ounce, including
    today's change and percentage change from the previous close."""

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
            f"Day range: ${data['low_price']:.2f} – ${data['high_price']:.2f}. "
            f"24K per gram: ${data['price_gram_24k']:.2f}."
        )
    except requests.RequestException as e:
        return f"Error fetching gold price: {e}"
    except (KeyError, TypeError) as e:
        return f"Error parsing gold price data: {e}"


@mcp.tool
def convert_currency(amount: float, to_currency: str) -> str:
    """Convert a USD amount to another currency using live exchange rates.

    Args:
        amount: The amount in USD to convert.
        to_currency: The target currency code (e.g., EUR, GBP, AUD, JPY, CAD).
    """
    url = f"https://api.frankfurter.dev/v1/latest?base=USD&symbols={to_currency.upper()}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        rate = data["rates"][to_currency.upper()]
        converted = amount * rate
        return (
            f"USD {amount:,.2f} = {to_currency.upper()} {converted:,.2f} "
            f"(rate: 1 USD = {rate:.4f} {to_currency.upper()})"
        )
    except requests.RequestException as e:
        return f"Error fetching exchange rate: {e}"
    except (KeyError, TypeError) as e:
        return f"Error parsing exchange rate data: {e}"


@mcp.tool
def calculate_portfolio(ounces: float, price_per_ounce: float) -> str:
    """Calculate the total value of a gold holding.

    Args:
        ounces: How many troy ounces of gold the user owns.
        price_per_ounce: The current gold price in USD per troy ounce.
    """
    total = ounces * price_per_ounce
    return (
        f"Portfolio: {ounces} troy oz × ${price_per_ounce:.2f}/oz "
        f"= ${total:,.2f} USD"
    )


if __name__ == "__main__":
    mcp.run()