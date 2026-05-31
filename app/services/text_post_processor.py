"""
OCR FinSight - Text Post-Processor
Fix common OCR errors after extraction.
"""

import re


class TextPostProcessor:
    """Post-process OCR text to fix common errors."""

    CHAR_CORRECTIONS = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1',
        'S': '5', 'Z': '2', 'B': '8',
    }

    WORD_CORRECTIONS = {
        'T0TAL': 'TOTAL', 'TOTA1': 'TOTAL', 'T0TA1': 'TOTAL',
        'TUNA1': 'TUNAI', 'KEMBA1I': 'KEMBALI', 'KEMBAL1': 'KEMBALI',
        'SUBT0TAL': 'SUBTOTAL', 'SUBTO1AL': 'SUBTOTAL',
        'PAJA<': 'PAJAK', 'DISK0N': 'DISKON', 'DISC0UNT': 'DISCOUNT',
        'CA5H': 'CASH', 'CHANG3': 'CHANGE', 'DAT3': 'DATE',
        'INV0ICE': 'INVOICE', 'INVO1CE': 'INVOICE',
        'RECE1PT': 'RECEIPT', 'RECE!PT': 'RECEIPT',
        'Q1Y': 'QTY', 'QUANT1TY': 'QUANTITY',
        'PR1CE': 'PRICE', 'PUCE': 'PRICE',
        'AM0UNT': 'AMOUNT', 'AMO1NT': 'AMOUNT',
        'DESCR1PTION': 'DESCRIPTION', 'DESCRIPT10N': 'DESCRIPTION',
        'CASH1ER': 'CASHIER', 'TE1P': 'TELP', 'TE1': 'TEL',
        'EMA1L': 'EMAIL', 'NPW?': 'NPWP',
    }

    def fix_numbers(self, text: str) -> str:
        def replace_in_number(match):
            num_str = match.group(0)
            for wrong, correct in self.CHAR_CORRECTIONS.items():
                num_str = num_str.replace(wrong, correct)
            return num_str

        patterns = [
            r'\d+[OoIlSZB\d.,]+',
            r'[OoIlSZB\d.,]+\d+',
            r'Rp\s*[OoIlSZB\d.,]+',
            r'\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4}',
        ]
        for pattern in patterns:
            text = re.sub(pattern, replace_in_number, text)
        return text

    def fix_words(self, text: str) -> str:
        for wrong, correct in self.WORD_CORRECTIONS.items():
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
        return text

    def fix_currency(self, text: str) -> str:
        text = re.sub(r'Rp\s*([OoIlSZB\d.,]+)', lambda m: 'Rp ' + self.fix_numbers(m.group(1)), text)
        text = re.sub(r'\$\s*([OoIlSZB\d.,]+)', lambda m: '$' + self.fix_numbers(m.group(1)), text)
        return text

    def fix_date(self, text: str) -> str:
        def fix_date_match(match):
            return match.group(0).replace('O', '0').replace('o', '0')
        return re.sub(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', fix_date_match, text)

    def process(self, text: str) -> str:
        if not text:
            return text
        text = self.fix_words(text)
        text = self.fix_numbers(text)
        text = self.fix_currency(text)
        text = self.fix_date(text)
        return re.sub(r'\s+', ' ', text).strip()

    def process_lines(self, lines: list[dict]) -> list[dict]:
        for line in lines:
            if 'text' in line:
                line['text'] = self.process(line['text'])
        return lines


_post_processor = None


def get_post_processor() -> TextPostProcessor:
    global _post_processor
    if _post_processor is None:
        _post_processor = TextPostProcessor()
    return _post_processor
