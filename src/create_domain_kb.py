"""Script to create and populate domain knowledge bases"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from domains.domain_registry import init_domains
from domains.teaching_abc import TeachingABCDomain
from domains.product_pricing import ProductPricingDomain


def create_all_domain_knowledge_bases():
    """Create and populate all domain knowledge bases"""
    
    print("=" * 60)
    print("CREATING DOMAIN KNOWLEDGE BASES")
    print("=" * 60)
    
    # Initialize domains
    print("\n[1/4] Initializing domains...")
    registry = init_domains("knowledge_bases")
    
    # Show created domains
    print(f"\n[2/4] Created {len(registry.get_all_domains())} domains:")
    for domain_id, domain in registry.get_all_domains().items():
        info = domain.get_info()
        print(f"   • {domain_id}: {info['name']} - {info['knowledge_count']} items")
    
    # Add custom knowledge to product pricing
    print("\n[3/4] Adding additional knowledge...")
    
    pricing_domain = registry.get_domain("product_pricing")
    if pricing_domain:
        pricing_domain.add_knowledge(
            "MacBook Air M2: $999 (8GB RAM, 256GB SSD), $1199 (8GB RAM, 512GB SSD). Best for students and professionals.",
            "apple_products"
        )
        pricing_domain.add_knowledge(
            "Dell XPS 13: $899 (Intel i5, 8GB RAM, 256GB SSD), $1099 (Intel i7, 16GB RAM, 512GB SSD). Best Windows ultrabook.",
            "dell_products"
        )
        print("   ✅ Added MacBook and Dell XPS pricing")
    
    teaching_domain = registry.get_domain("teaching_abc")
    if teaching_domain:
        teaching_domain.add_knowledge(
            "The letter D makes the 'duh' sound like in 'dog', 'duck', and 'dinosaur'. D is the 4th letter of the alphabet.",
            "alphabet"
        )
        teaching_domain.add_knowledge(
            "Words that start with C: cat, car, cup, cake, crayon. Practice saying 'kuh kuh cat'!",
            "vocabulary"
        )
        print("   ✅ Added more alphabet teaching content")
    
    # Summary
    print("\n[4/4] Knowledge bases ready!")
    print("\n📁 Location: knowledge_bases/")
    print("\nTo use these domains:")
    print("   python main.py")
    print("   Then type: /mode teaching_abc")
    print("   Or ask: What sound does the letter A make?")


if __name__ == "__main__":
    create_all_domain_knowledge_bases()