"""
Classifier Post-Correction Layer
=================================
Fix common classification errors using rule-based heuristics.

This layer runs AFTER the ML classifier to fix systematic errors:
1. GRAND_TOTAL vs SUBTOTAL disambiguation
2. ITEM_PRICE/QTY vs GRAND_TOTAL disambiguation
3. TAX vs SERVICE_CHARGE disambiguation
4. Context-aware correction (position, text patterns)

Usage:
    corrector = ClassifierCorrector()
    corrected_lines = corrector.correct(classified_lines)
"""

import re
from typing import List, Dict


class ClassifierCorrector:
    """Rule-based post-processor to fix common classifier mistakes."""
    
    def __init__(self):
        # Grand total keywords (strongest signals)
        self.grand_total_keywords = [
            'grand total', 'total bayar', 'total amount', 'total pembayaran',
            'total akhir', 'nett total', 'net total', 'amount due', 'balance due',
            'total due', 'total tagihan', 'jumlah bayar'
        ]
        
        # Subtotal keywords
        self.subtotal_keywords = [
            'subtotal', 'sub total', 'sub-total', 'jumlah', 'sub ttl',
            'amount', 'total harga', 'total item', 'total barang',
            'exclude', 'before', 'sebelum'  # "before tax", "exclude gst"
        ]
        
        # Tax keywords
        self.tax_keywords = [
            'tax', 'pajak', 'ppn', 'gst', 'vat', 'pb1', 'sst',
            'add gst', 'add tax', 'govt tax', 'sales tax'
        ]
        
        # Service charge keywords
        self.service_keywords = [
            'service', 'servis', 'service charge', 'service fee',
            'layanan', 'pelayanan', 'svc', 'svc charge'
        ]
        
        # Discount keywords
        self.discount_keywords = [
            'discount', 'diskon', 'potongan', 'disc', 'voucher',
            'promo', 'cashback', 'rebate', 'kupon'
        ]
    
    def correct(self, lines: List[Dict]) -> List[Dict]:
        """Apply rule-based corrections to classified lines.
        
        Args:
            lines: List of classified lines with 'text', 'predicted_class', etc.
            
        Returns:
            Corrected lines (modified in place + returned)
        """
        if not lines:
            return lines
        
        # Sort by y_min for position-aware logic
        sorted_lines = sorted(enumerate(lines), key=lambda x: x[1].get('y_min', 0))
        
        # Pass 1: Keyword-based strong corrections
        for idx, line in sorted_lines:
            text = line['text'].lower()
            current_class = line['predicted_class']
            
            # Strong keyword matches override classifier
            # But only if the keyword is VERY specific
            if self._has_keyword(text, self.grand_total_keywords):
                line['predicted_class'] = 'GRAND_TOTAL'
                line['correction_reason'] = 'grand_total_keyword'
            
            elif self._has_keyword(text, self.subtotal_keywords):
                # Only override if VERY sure (has "sub" prefix)
                if any(k in text for k in ['subtotal', 'sub total', 'sub-total', 'sub ttl']):
                    line['predicted_class'] = 'SUBTOTAL'
                    line['correction_reason'] = 'subtotal_keyword'
            
            elif self._has_keyword(text, self.tax_keywords) and not self._has_keyword(text, self.service_keywords):
                # Only override if current class is NOT already correct
                if current_class not in ['TAX', 'GRAND_TOTAL', 'SUBTOTAL']:
                    line['predicted_class'] = 'TAX'
                    line['correction_reason'] = 'tax_keyword'
            
            elif self._has_keyword(text, self.service_keywords):
                # Only override if NOT already TAX (service tax = TAX)
                if current_class not in ['SERVICE_CHARGE', 'TAX']:
                    line['predicted_class'] = 'SERVICE_CHARGE'
                    line['correction_reason'] = 'service_keyword'
            
            elif self._has_keyword(text, self.discount_keywords):
                if current_class != 'DISCOUNT':
                    line['predicted_class'] = 'DISCOUNT'
                    line['correction_reason'] = 'discount_keyword'
        
        # Pass 2: Context-aware disambiguation (OPTIONAL - only for ambiguous cases)
        # Disable aggressive corrections for now
        # self._fix_grand_total_vs_subtotal(sorted_lines)
        # self._fix_grand_total_vs_item_price(sorted_lines)
        # self._fix_tax_vs_service_charge(sorted_lines)
        
        return lines
    
    def _has_keyword(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords."""
        return any(kw in text for kw in keywords)
    
    def _fix_grand_total_vs_subtotal(self, sorted_lines: List[tuple]):
        """Disambiguate GRAND_TOTAL vs SUBTOTAL using context.
        
        Rules:
        1. GRAND_TOTAL usually appears AFTER subtotal/tax/service_charge
        2. GRAND_TOTAL is typically at y > 0.7 (bottom 30% of receipt)
        3. If multiple GRAND_TOTAL candidates, pick the one with highest y_min
        4. SUBTOTAL appears BEFORE tax/service/discount lines
        """
        # Find all total-like lines
        total_candidates = [
            (idx, line) for idx, line in sorted_lines
            if line['predicted_class'] in ['GRAND_TOTAL', 'SUBTOTAL']
            and 'correction_reason' not in line  # Don't override keyword corrections
        ]
        
        if len(total_candidates) < 2:
            return  # No ambiguity
        
        # Find tax/service/discount lines (markers that come before grand total)
        marker_y_positions = [
            line.get('y_min', 0) for _, line in sorted_lines
            if line['predicted_class'] in ['TAX', 'SERVICE_CHARGE', 'DISCOUNT']
        ]
        
        if not marker_y_positions:
            # No markers: ONLY correct if very confident based on position
            # Last total candidate at y > 0.75 = GRAND_TOTAL
            for i, (idx, line) in enumerate(total_candidates):
                line_y = line.get('y_min', 0)
                if i == len(total_candidates) - 1 and line_y > 0.75:
                    line['predicted_class'] = 'GRAND_TOTAL'
                    line['correction_reason'] = 'position_last_total'
        else:
            # Use markers: only correct if clearly separated
            max_marker_y = max(marker_y_positions)
            
            for idx, line in total_candidates:
                line_y = line.get('y_min', 0)
                
                # Only correct if there's a clear gap (> 0.03 or ~5% of image)
                if line_y > max_marker_y + 0.03:
                    line['predicted_class'] = 'GRAND_TOTAL'
                    line['correction_reason'] = 'position_after_markers'
                elif line_y < max_marker_y - 0.03:
                    line['predicted_class'] = 'SUBTOTAL'
                    line['correction_reason'] = 'position_before_markers'
    
    def _fix_grand_total_vs_item_price(self, sorted_lines: List[tuple]):
        """Fix ITEM_PRICE/QTY misclassified as GRAND_TOTAL.
        
        Rules:
        1. If classified as GRAND_TOTAL but y < 0.6 → likely ITEM_PRICE
        2. If classified as GRAND_TOTAL but x_min < 0.4 → likely ITEM_DESC
        3. If many ITEM_PRICE lines at similar y → not GRAND_TOTAL
        """
        # Find GRAND_TOTAL candidates
        grand_totals = [
            (idx, line) for idx, line in sorted_lines
            if line['predicted_class'] == 'GRAND_TOTAL'
        ]
        
        # Find item zone (where most ITEM lines are)
        item_y_positions = [
            line.get('y_min', 0) for _, line in sorted_lines
            if line['predicted_class'] in ['ITEM_DESC', 'ITEM_PRICE/QTY']
        ]
        
        if not item_y_positions:
            return
        
        item_zone_start = min(item_y_positions)
        item_zone_end = max(item_y_positions)
        
        for idx, line in grand_totals:
            line_y = line.get('y_min', 0)
            line_x = line.get('x_min', 0)
            
            # If GRAND_TOTAL is in item zone → likely misclassified
            if item_zone_start <= line_y <= item_zone_end:
                # Check if it's in price column (right side)
                if line_x >= 0.5:
                    line['predicted_class'] = 'ITEM_PRICE/QTY'
                    line['correction_reason'] = 'in_item_zone_price_column'
                else:
                    # Left side → could be item name
                    line['predicted_class'] = 'ITEM_DESC'
                    line['correction_reason'] = 'in_item_zone_name_column'
    
    def _fix_tax_vs_service_charge(self, sorted_lines: List[tuple]):
        """Disambiguate TAX vs SERVICE_CHARGE.
        
        Rules:
        1. If text contains "service" → SERVICE_CHARGE
        2. If text contains "tax", "gst", "ppn" → TAX
        3. If amount is ~6-10% of subtotal → likely TAX
        4. If amount is ~5-15% of subtotal → likely SERVICE_CHARGE
        """
        # Find subtotal for percentage calculation
        subtotal_value = 0.0
        for _, line in sorted_lines:
            if line['predicted_class'] == 'SUBTOTAL':
                text = line['text'].replace(' ', '')
                # Extract number
                numbers = re.findall(r'\d{1,3}(?:[,.]\d{3})+|\d+', text)
                if numbers:
                    try:
                        val = float(numbers[-1].replace(',', '').replace('.', ''))
                        # Handle decimal: if val > 100000, assume no decimal
                        if val < 100000:
                            val = float(numbers[-1].replace(',', '.'))
                        subtotal_value = val
                        break
                    except:
                        pass
        
        if subtotal_value <= 0:
            return  # Can't validate by percentage
        
        # Check TAX and SERVICE_CHARGE lines
        for idx, line in sorted_lines:
            if line['predicted_class'] not in ['TAX', 'SERVICE_CHARGE']:
                continue
            
            text = line['text']
            text_lower = text.lower()
            
            # Extract amount
            numbers = re.findall(r'\d{1,3}(?:[,.]\d{3})+|\d+', text.replace(' ', ''))
            if not numbers:
                continue
            
            try:
                amount = float(numbers[-1].replace(',', '').replace('.', ''))
                if amount < 100000:
                    amount = float(numbers[-1].replace(',', '.'))
            except:
                continue
            
            # Calculate percentage of subtotal
            if amount > 0:
                percentage = (amount / subtotal_value) * 100
                
                # Tax is usually 6-11% (Malaysia GST 6%, Indonesia PPN 11%)
                # Service charge is usually 5-10%
                if 5 <= percentage <= 11:
                    if 'service' in text_lower or 'servis' in text_lower:
                        line['predicted_class'] = 'SERVICE_CHARGE'
                        line['correction_reason'] = 'service_keyword_with_percentage'
                    elif any(k in text_lower for k in ['tax', 'gst', 'ppn', 'vat', 'pb1']):
                        line['predicted_class'] = 'TAX'
                        line['correction_reason'] = 'tax_keyword_with_percentage'


def apply_corrections(lines: List[Dict]) -> List[Dict]:
    """Convenience function to apply corrections.
    
    Usage:
        from src.classifier_corrector import apply_corrections
        corrected = apply_corrections(classified_lines)
    """
    corrector = ClassifierCorrector()
    return corrector.correct(lines)
