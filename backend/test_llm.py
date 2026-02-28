import asyncio
import sys
import json
sys.path.insert(0, ".")
from config import WEIGHT_RATING, WEIGHT_REVIEWS, WEIGHT_PRICE, WEIGHT_OFFERS, WEIGHT_NON_SPONSORED
from llm.explainer import explain_and_select_product
from scoring.engine import score_products

# Mock products
mock_products = [
    {
        "title": "Super Premium Earbuds - ₹2500 (Better Rating)",
        "price": 2500,
        "rating": 4.9,
        "review_count": 50000,
        "sponsored": False,
        "offers": ""
    },
    {
        "title": "Good Earbuds - ₹1800",
        "price": 1800,
        "rating": 4.5,
        "review_count": 20000,
        "sponsored": False,
        "offers": "5% off"
    },
    {
        "title": "Budget Earbuds - ₹1400",
        "price": 1400,
        "rating": 4.1,
        "review_count": 10000,
        "sponsored": False,
        "offers": ""
    }
]

async def run_test():
    print("Scoring products to simulate typical top 5...")
    scored = score_products(mock_products)
    
    print("\nDeterministic Ranking:")
    for idx, p in enumerate(scored):
        print(f"[{idx}] {p['title']} - Score {p['score']}")
    
    print("\n--- Testing Strict LLM Constraints ---")
    query = "earbuds under 2000"
    print(f"User Query: '{query}'")
    
    print("Calling explain_and_select_product...")
    idx, exp, rev = await explain_and_select_product(query, scored)
    
    print(f"\nResult:")
    print(f"LLM picked Index: {idx} -> {scored[idx]['title']}")
    print(f"Explanation:\n{exp}")
    print(f"Review Summary:\n{rev}")
    
    if scored[idx]['price'] > 2000:
        print("\n❌ FAILURE: LLM selected a product over the price limit!")
    else:
        print("\n✅ SUCCESS: LLM respected the price constraint.")

if __name__ == "__main__":
    asyncio.run(run_test())
