"""
OCR FinSight - Text Post-Processor
Fix common OCR errors after extraction.
"""

import re


class TextPostProcessor:
    """Post-process OCR text to fix common errors."""
    
    # Common OCR character confusions
    CHAR_CORRECTIONS = {
        # Number/Letter confusions in numeric contexts
        'O': '0',  # O → 0 in numbers
        'o': '0',  # o → 0 in numbers
        'I': '1',  # I → 1 in numbers
        'l': '1',  # l → 1 in numbers
        'S': '5',  # S → 5 in numbers
        'Z': '2',  # Z → 2 in numbers
        'B': '8',  # B → 8 in numbers
    }
    
    # Common word corrections
    WORD_CORRECTIONS = {
        # Common receipt words
        'T0TAL': 'TOTAL',
        'TOTA1': 'TOTAL',
        'T0TA1': 'TOTAL',
        'TUNAI': 'TUNAI',
        'TUNA1': 'TUNAI',
        'KEMBALI': 'KEMBALI',
        'KEMBA1I': 'KEMBALI',
        'KEMBAL1': 'KEMBALI',
        'SUBTOTAL': 'SUBTOTAL',
        'SUBT0TAL': 'SUBTOTAL',
        'SUBTO1AL': 'SUBTOTAL',
        'PAJAK': 'PAJAK',
        'PAJA<': 'PAJAK',
        'DISKON': 'DISKON',
        'DISK0N': 'DISKON',
        'DISC0UNT': 'DISCOUNT',
        'CASH': 'CASH',
        'CA5H': 'CASH',
        'CHANGE': 'CHANGE',
        'CHANG3': 'CHANGE',
        'DATE': 'DATE',
        'DAT3': 'DATE',
        'INVOICE': 'INVOICE',
        'INV0ICE': 'INVOICE',
        'INVO1CE': 'INVOICE',
        'RECEIPT': 'RECEIPT',
        'RECE1PT': 'RECEIPT',
        'RECE!PT': 'RECEIPT',
        'QTY': 'QTY',
        'Q1Y': 'QTY',
        'QUANTITY': 'QUANTITY',
        'QUANT1TY': 'QUANTITY',
        'PRICE': 'PRICE',
        'PR1CE': 'PRICE',
        'PUCE': 'PRICE',
        'AMOUNT': 'AMOUNT',
        'AM0UNT': 'AMOUNT',
        'AMO1NT': 'AMOUNT',
        'ITEM': 'ITEM',
        'DESCRIPTION': 'DESCRIPTION',
        'DESCR1PTION': 'DESCRIPTION',
        'DESCRIPT10N': 'DESCRIPTION',
        'CASHIER': 'CASHIER',
        'CASH1ER': 'CASHIER',
        'TELP': 'TELP',
        'TE1P': 'TELP',
        'TEL': 'TEL',
        'TE1': 'TEL',
        'EMAIL': 'EMAIL',
        'EMA1L': 'EMAIL',
        'NPWP': 'NPWP',
        'NPW?': 'NPWP',
    }
    
    def __init__(self):
        """Initialize post-processor."""
        pass
    
    def fix_numbers(self, text: str) -> str:
        """Fix common OCR errors in numbers.
        
        Examples:
            "Rp 1O.OOO" → "Rp 10.000"
            "Tel: O7-388" → "Tel: 07-388"
            "Invoice: 1O3O765" → "Invoice: 1030765"
        """
        # Pattern: numbers with O/o/I/l/S/Z/B
        def replace_in_number(match):
            num_str = match.group(0)
            for wrong, correct in self.CHAR_CORRECTIONS.items():
                num_str = num_str.replace(wrong, correct)
            return num_str
        
        # Match number patterns (with possible letter confusions)
        patterns = [
            r'\d+[OoIlSZB\d.,]+',  # Numbers with confusions
            r'[OoIlSZB\d.,]+\d+',  # Confusions with numbers
            r'Rp\s*[OoIlSZB\d.,]+',  # Currency
            r'\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4}',  # Dates
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, replace_in_number, text)
        
        return text
    
    def fix_words(self, text: str) -> str:
        """Fix common OCR errors in words."""
        # Case-insensitive word replacement
        for wrong, correct in self.WORD_CORRECTIONS.items():
            # Match whole words only
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
        
        return text
    
    def fix_currency(self, text: str) -> str:
        """Fix currency formatting.
        
        Examples:
            "Rp 1O.OOO" → "Rp 10.000"
            "Rp1O,OOO" → "Rp 10,000"
        """
        # Fix Rp formatting
        text = re.sub(r'Rp\s*([OoIlSZB\d.,]+)', lambda m: 'Rp ' + self.fix_numbers(m.group(1)), text)
        
        # Fix $ formatting
        text = re.sub(r'\$\s*([OoIlSZB\d.,]+)', lambda m: '$' + self.fix_numbers(m.group(1)), text)
        
        return text
    
    def fix_date(self, text: str) -> str:
        """Fix date formatting.
        
        Examples:
            "O2/O1/2O19" → "02/01/2019"
            "15-O1-2O19" → "15-01-2019"
        """
        # Pattern: DD/MM/YYYY or DD-MM-YYYY
        def fix_date_match(match):
            date_str = match.group(0)
            # Replace O with 0 in dates
            date_str = date_str.replace('O', '0').replace('o', '0')
            return date_str
        
        text = re.sub(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', fix_date_match, text)
        
        return text
    
    def process(self, text: str) -> str:
        """Apply all post-processing fixes.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        # Apply fixes in order
        text = self.fix_words(text)
        text = self.fix_numbers(text)
        text = self.fix_currency(text)
        text = self.fix_date(text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_lines(self, lines: list[dict]) -> list[dict]:
        """Process multiple OCR lines.
        
        Args:
            lines: List of OCR line dicts with 'text' key
            
        Returns:
            Lines with corrected text
        """
        for line in lines:
            if 'text' in line:
                line['text'] = self.process(line['text'])
        
        return lines


# Singleton instance
_post_processor = None

def get_post_processor() -> TextPostProcessor:
    """Get singleton post-processor instance."""
    global _post_processor
    if _post_processor is None:
        _post_processor = TextPostProcessor()
    return _post_processor
