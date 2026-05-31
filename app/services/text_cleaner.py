"""
OCR FinSight - Text Cleaner
Clean and normalize OCR output per field type.
"""

import re


class OCRTextCleaner:

    WORD_CORRECTIONS = {
        'k0ta': 'kota', 'c0': 'co', 't0k0': 'toko', 'c1rcle': 'circle',
        'c1ty': 'city', 'ma11': 'mall', 'p1aza': 'plaza',
        'tota1': 'total', 't0tal': 'total', 'tange1': 'tanggal',
        'a1amat': 'alamat', 'te1p': 'telp', 'n0': 'no', 'n0.': 'no.',
        'ca5h': 'cash', 'ca5hier': 'cashier', 'cust0mer': 'customer',
        'rece1pt': 'receipt', 'inv0ice': 'invoice', 'b1ll': 'bill',
        'qfy': 'qty', 'q7y': 'qty', 'puce': 'price', 'pr1ce': 'price',
        'amoynt': 'amount', 'am0unt': 'amount', 'addres': 'address',
        'subt0tal': 'subtotal', 'disc0unt': 'discount', 'diskon': 'discount',
        'kemba1i': 'kembali', 'tuna1': 'tunai', 'paja': 'pajak',
        'kemba1ian': 'kembalian', 'jumlah': 'jumlah', 'jum1ah': 'jumlah',
        'bayar': 'bayar', 'bay4r': 'bayar', 'terima': 'terima', 'kasih': 'kasih',
    }

    @staticmethod
    def clean_text(text: str, context: str = 'general') -> str:
        if not text:
            return text
        text = text.strip()
        if context == 'store':
            text = OCRTextCleaner._clean_store_name(text)
        elif context == 'address':
            text = OCRTextCleaner._clean_address(text)
        elif context == 'item':
            text = OCRTextCleaner._clean_item_name(text)
        elif context == 'number':
            text = OCRTextCleaner._clean_number(text)
        else:
            text = OCRTextCleaner._clean_general(text)
        text = OCRTextCleaner._apply_word_corrections(text)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _clean_store_name(text: str) -> str:
        replacements = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '3': 'E', '4': 'A'}
        result = []
        for word in text.split():
            alpha = sum(c.isalpha() for c in word)
            digit = sum(c.isdigit() for c in word)
            if alpha > digit and digit > 0:
                for d, l in replacements.items():
                    word = word.replace(d, l)
            result.append(word)
        return ' '.join(result)

    @staticmethod
    def _clean_address(text: str) -> str:
        text = re.sub(r'\bn0\.?\s*', 'No. ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bj[l1]\.?\s*', 'Jl. ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bte[l1]p\.?\s*', 'Telp. ', text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _clean_item_name(text: str) -> str:
        return re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)

    @staticmethod
    def _clean_number(text: str) -> str:
        replacements = {'O': '0', 'o': '0', 'I': '1', 'l': '1',
                        'S': '5', 's': '5', 'B': '8', 'Z': '2', 'z': '2'}
        for letter, digit in replacements.items():
            text = text.replace(letter, digit)
        text = re.sub(r'(\d+),\.(\d+)', r'\1.\2', text)
        return re.sub(r'[^\d.,/\-]', '', text)

    @staticmethod
    def _clean_general(text: str) -> str:
        return re.sub(r'[^\x00-\x7F]+', '', text)

    @staticmethod
    def _apply_word_corrections(text: str) -> str:
        words = text.split()
        corrected = []
        for word in words:
            lower = word.lower()
            if lower in OCRTextCleaner.WORD_CORRECTIONS:
                fix = OCRTextCleaner.WORD_CORRECTIONS[lower]
                if word.isupper():
                    corrected.append(fix.upper())
                elif word[0].isupper():
                    corrected.append(fix.capitalize())
                else:
                    corrected.append(fix)
            else:
                corrected.append(word)
        return ' '.join(corrected)

    @staticmethod
    def clean_store(text: str) -> str:
        return OCRTextCleaner.clean_text(text, context='store')

    @staticmethod
    def clean_address(text: str) -> str:
        return OCRTextCleaner.clean_text(text, context='address')

    @staticmethod
    def clean_item(text: str) -> str:
        return OCRTextCleaner.clean_text(text, context='item')

    @staticmethod
    def clean_number(text: str) -> str:
        return OCRTextCleaner.clean_text(text, context='number')

    @staticmethod
    def clean_date(text: str) -> str:
        if not text:
            return text
        text = re.sub(r'(\d{1,2})\.(\d{2})\.(\d{2})', r'\1:\2:\3', text)
        text = re.sub(r'(\d{1,2}),(\d{2}),(\d{2})', r'\1:\2:\3', text)
        text = re.sub(r'^Date\.?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Time\.?\s*', '', text, flags=re.IGNORECASE)
        return text.strip()
