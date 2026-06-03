"""
Output Formatter for Receipt Extraction
========================================
Transform extraction result to clean, simple JSON format.

Target format:
{
    "success": true,
    "store": "Store Name",
    "date": "Jun 18, 2023",
    "items": [
        {"name": "Item Name", "qty": 1, "price": 29000.0}
    ],
    "total": 169400.0
}
"""

from typing import Dict, List, Any


class OutputFormatter:
    """Format extraction results to clean JSON."""
    
    @staticmethod
    def format_simple(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format to simple clean JSON.
        
        Args:
            extraction_result: Dict from ReceiptExtractor.extract()
            
        Returns:
            Clean formatted dict with: success, store, date, items, total
        """
        # Determine success (has store or items or total)
        has_content = bool(
            extraction_result.get('store') or 
            extraction_result.get('items') or 
            extraction_result.get('total', 0) > 0
        )
        
        # Clean items: remove duplicates and format
        items = []
        seen_items = set()
        
        for item in extraction_result.get('items', []):
            # Create unique key (name + price to detect duplicates)
            item_key = (item.get('name', '').strip().lower(), item.get('price', 0))
            
            if item_key in seen_items:
                continue
            
            seen_items.add(item_key)
            
            # Clean item name (remove extra spaces, normalize)
            cleaned_name = ' '.join(item.get('name', '').split())
            
            # Skip if name too short or invalid
            if len(cleaned_name) < 3:
                continue
            
            items.append({
                'name': cleaned_name,
                'qty': int(item.get('qty', 1)),
                'price': float(item.get('price', 0))
            })
        
        # Get total (prefer grand_total from totals dict)
        total = 0.0
        if 'totals' in extraction_result:
            total = float(extraction_result['totals'].get('grand_total', 0))
        if total == 0:
            total = float(extraction_result.get('total', 0))
        
        # Fallback: sum item prices if total not found
        if total == 0 and items:
            total = sum(item['price'] * item['qty'] for item in items)
        
        return {
            'success': has_content,
            'store': extraction_result.get('store', '').strip(),
            'date': extraction_result.get('date', '').strip(),
            'items': items,
            'total': total
        }
    
    @staticmethod
    def format_detailed(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format with additional details (subtotal, tax, etc).
        
        Args:
            extraction_result: Dict from ReceiptExtractor.extract()
            
        Returns:
            Detailed formatted dict
        """
        simple = OutputFormatter.format_simple(extraction_result)
        
        # Add detailed totals breakdown
        totals_detail = {}
        if 'totals' in extraction_result:
            totals = extraction_result['totals']
            totals_detail = {
                'subtotal': float(totals.get('subtotal', 0)),
                'tax': float(totals.get('tax', 0)),
                'service_charge': float(totals.get('service_charge', 0)),
                'discount': float(totals.get('discount', 0)),
                'grand_total': float(totals.get('grand_total', 0)),
            }
        
        # Add address if available
        address = extraction_result.get('address', '').strip()
        
        return {
            **simple,
            'address': address,
            'totals_breakdown': totals_detail,
        }


def format_extraction_result(result: Dict[str, Any], format_type: str = 'simple') -> Dict[str, Any]:
    """Convenience function to format extraction result.
    
    Args:
        result: Output from ReceiptExtractor.extract()
        format_type: 'simple' or 'detailed'
        
    Returns:
        Formatted dict
        
    Example:
        >>> from src.extractor import ReceiptExtractor
        >>> from src.output_formatter import format_extraction_result
        >>> 
        >>> extractor = ReceiptExtractor()
        >>> result = extractor.extract(classified_lines)
        >>> clean_result = format_extraction_result(result, 'simple')
        >>> print(clean_result)
        {
            "success": true,
            "store": "Kopi Nako",
            "date": "Jun 18, 2023",
            "items": [...],
            "total": 169400.0
        }
    """
    formatter = OutputFormatter()
    
    if format_type == 'detailed':
        return formatter.format_detailed(result)
    else:
        return formatter.format_simple(result)
