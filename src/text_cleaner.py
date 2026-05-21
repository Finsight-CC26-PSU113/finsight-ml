"""
OCR FinSight - Text Cleaner
Membersihkan dan memperbaiki hasil OCR yang sering salah.
"""

import re


class OCRTextCleaner:
    """Clean common OCR errors dan normalize text."""
    
    # Common OCR misreads
    CHAR_REPLACEMENTS = {
        # Digit → Letter
        '0': 'O',  # Zero → O (dalam konteks huruf)
        '1': 'I',  # One → I (dalam konteks huruf)
        '5': 'S',  # Five → S (dalam konteks huruf)
        '8': 'B',  # Eight → B (dalam konteks huruf)
        
        # Letter → Digit (akan dihandle by context)
        'O': '0',  # O → Zero (dalam konteks angka)
        'I': '1',  # I → One (dalam konteks angka)
        'l': '1',  # lowercase L → One
        'S': '5',  # S → Five (dalam konteks angka)
        'B': '8',  # B → Eight (dalam konteks angka)
    }
    
    # Common word corrections (case-insensitive)
    WORD_CORRECTIONS = {
        # Store names
        'k0ta': 'kota',
        'c0': 'co',
        't0k0': 'toko',
        'c1rcle': 'circle',
        'c1ty': 'city',
        'ma11': 'mall',
        'p1aza': 'plaza',
        
        # Common words
        'tota1': 'total',
        't0tal': 'total',
        'tange1': 'tanggal',
        'tangg4l': 'tanggal',
        'a1amat': 'alamat',
        'te1p': 'telp',
        'te1epon': 'telepon',
        'n0': 'no',
        'n0.': 'no.',
        'jl.': 'jalan',
        'j1.': 'jalan',
        
        # Receipt terms
        'ca5h': 'cash',
        'ca5hier': 'cashier',
        'cust0mer': 'customer',
        'cust0ner': 'customer',
        'rece1pt': 'receipt',
        'inv0ice': 'invoice',
        'invcice': 'invoice',
        'b1ll': 'bill',
        
        # Numbers in words
        'qty': 'qty',
        'qfy': 'qty',
        'q7y': 'qty',
        
        # Common OCR typos (EXPANDED)
        'persor': 'person',
        'descpton': 'description',
        'descripion': 'description',
        'puce': 'price',
        'pr1ce': 'price',
        'amoynt': 'amount',
        'am0unt': 'amount',
        'masal': 'masai',
        'masa1': 'masai',
        'addres': 'address',
        'addr3ss': 'address',
        'te1': 'tel',
        'fax': 'fax',
        'emai1': 'email',
        'ema1l': 'email',
        'subt0tal': 'subtotal',
        'sub-t0tal': 'subtotal',
        'disc0unt': 'discount',
        'disc': 'discount',
        'diskon': 'discount',
        'p0tongan': 'potongan',
        'kemba1i': 'kembali',
        'kemba1ian': 'kembalian',
        'tuna1': 'tunai',
        'tun4i': 'tunai',
        'paja': 'pajak',
        'paj4k': 'pajak',
        'ppn': 'ppn',
        'gst': 'gst',
        'vat': 'vat',
        'sa1es': 'sales',
        'sa1e': 'sale',
        'ite': 'item',
        'it3m': 'item',
        'prod': 'product',
        'pr0duct': 'product',
        'barang': 'barang',
        'bar4ng': 'barang',
        'jumlah': 'jumlah',
        'jum1ah': 'jumlah',
        'grand': 'grand',
        'gr4nd': 'grand',
        'tota': 'total',
        't0ta': 'total',
        'rounding': 'rounding',
        'r0unding': 'rounding',
        'change': 'change',
        'ch4nge': 'change',
        'paid': 'paid',
        'pa1d': 'paid',
        'bayar': 'bayar',
        'bay4r': 'bayar',
        'card': 'card',
        'c4rd': 'card',
        'visa': 'visa',
        'v1sa': 'visa',
        'master': 'master',
        'mast3r': 'master',
        'approval': 'approval',
        'appr0val': 'approval',
        'code': 'code',
        'c0de': 'code',
        'thank': 'thank',
        'th4nk': 'thank',
        'terima': 'terima',
        'ter1ma': 'terima',
        'kasih': 'kasih',
        'kas1h': 'kasih',
    }
    
    @staticmethod
    def clean_text(text: str, context: str = 'general') -> str:
        """Clean OCR text based on context.
        
        Args:
            text: Raw OCR text
            context: 'store', 'address', 'item', 'number', 'general'
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        original = text
        
        # 1. Basic cleanup
        text = text.strip()
        
        # 2. Context-specific cleaning
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
        
        # 3. Word-level corrections
        text = OCRTextCleaner._apply_word_corrections(text)
        
        # 4. Remove excessive spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def _clean_store_name(text: str) -> str:
        """Clean store name - prioritize letters over digits."""
        # Replace common digit errors in store names
        replacements = {
            '0': 'O',
            '1': 'I',
            '5': 'S',
            '8': 'B',
            '3': 'E',
            '4': 'A',
        }
        
        result = []
        words = text.split()
        
        for word in words:
            # If word is mostly letters, replace digits
            alpha_count = sum(c.isalpha() for c in word)
            digit_count = sum(c.isdigit() for c in word)
            
            if alpha_count > digit_count and digit_count > 0:
                # Replace digits with likely letters
                cleaned = word
                for digit, letter in replacements.items():
                    cleaned = cleaned.replace(digit, letter)
                result.append(cleaned)
            else:
                result.append(word)
        
        return ' '.join(result)
    
    @staticmethod
    def _clean_address(text: str) -> str:
        """Clean address text."""
        # Fix common address patterns
        text = re.sub(r'\bn0\.?\s*', 'No. ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bj1\.?\s*', 'Jl. ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bjl\.?\s*', 'Jl. ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bte1p\.?\s*', 'Telp. ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bte1\.?\s*', 'Tel. ', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def _clean_item_name(text: str) -> str:
        """Clean item name - keep as is mostly."""
        # Remove leading/trailing special chars
        text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)
        return text
    
    @staticmethod
    def _clean_number(text: str) -> str:
        """Clean number - prioritize digits over letters."""
        # Replace common letter errors in numbers
        replacements = {
            'O': '0',
            'o': '0',
            'I': '1',
            'l': '1',
            'S': '5',
            's': '5',
            'B': '8',
            'Z': '2',
            'z': '2',
            'D': '0',  # 193D0 → 19300
            'u': '0',  # 0.u0 → 0.00
        }
        
        result = text
        for letter, digit in replacements.items():
            result = result.replace(letter, digit)
        
        # Fix common number format errors
        result = re.sub(r'(\d+),\.(\d+)', r'\1.\2', result)  # 193,.00 → 193.00
        result = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1\2.\3', result)  # 1.23.45 → 123.45
        
        # Remove non-numeric chars except . , /
        result = re.sub(r'[^\d.,/\-]', '', result)
        
        return result
    
    @staticmethod
    def _clean_general(text: str) -> str:
        """General cleaning."""
        # Remove weird unicode chars
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text
    
    @staticmethod
    def _apply_word_corrections(text: str) -> str:
        """Apply word-level corrections."""
        words = text.split()
        corrected = []
        
        for word in words:
            # Check lowercase version
            lower = word.lower()
            if lower in OCRTextCleaner.WORD_CORRECTIONS:
                # Preserve original case pattern
                correction = OCRTextCleaner.WORD_CORRECTIONS[lower]
                if word.isupper():
                    corrected.append(correction.upper())
                elif word[0].isupper():
                    corrected.append(correction.capitalize())
                else:
                    corrected.append(correction)
            else:
                corrected.append(word)
        
        return ' '.join(corrected)
    
    @staticmethod
    def clean_store(text: str) -> str:
        """Shortcut for cleaning store name."""
        return OCRTextCleaner.clean_text(text, context='store')
    
    @staticmethod
    def clean_address(text: str) -> str:
        """Shortcut for cleaning address."""
        return OCRTextCleaner.clean_text(text, context='address')
    
    @staticmethod
    def clean_item(text: str) -> str:
        """Shortcut for cleaning item name."""
        return OCRTextCleaner.clean_text(text, context='item')
    
    @staticmethod
    def clean_number(text: str) -> str:
        """Shortcut for cleaning number."""
        return OCRTextCleaner.clean_text(text, context='number')
    
    @staticmethod
    def clean_date(text: str) -> str:
        """Clean date text - fix common OCR errors in dates."""
        if not text:
            return text
        
        # Fix time punctuation errors
        # 11.05.16 → 11:05:16
        text = re.sub(r'(\d{1,2})\.(\d{2})\.(\d{2})', r'\1:\2:\3', text)
        # 11,05,16 → 11:05:16
        text = re.sub(r'(\d{1,2}),(\d{2}),(\d{2})', r'\1:\2:\3', text)
        # 11;05;16 → 11:05:16
        text = re.sub(r'(\d{1,2});(\d{2});(\d{2})', r'\1:\2:\3', text)
        
        # Fix year concatenation: 201817 → 2018 17
        text = re.sub(r'(\d{4})(\d{2})', r'\1 \2', text)
        
        # Remove "Date." prefix
        text = re.sub(r'^Date\.?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^Time\.?\s*', '', text, flags=re.IGNORECASE)
        
        # Fix common typos in dates
        text = re.sub(r'1i', '11', text)  # 20/1i/2017 → 20/11/2017
        text = re.sub(r'0i', '01', text)  # 0i/12/2017 → 01/12/2017
        
        # Fix weird time formats: 2247:14 → 22:47:14
        text = re.sub(r'(\d{2})(\d{2}):(\d{2})', r'\1:\2:\3', text)
        
        # Fix comma in time: 4.40,40 → 4:40:40
        text = re.sub(r'(\d{1,2})\.(\d{2}),(\d{2})', r'\1:\2:\3', text)
        
        return text.strip()


# Quick test
if __name__ == "__main__":
    cleaner = OCRTextCleaner()
    
    print("Store name tests:")
    print(f"  K0ta → {cleaner.clean_store('K0ta')}")
    print(f"  C1RCLE FRESH MART → {cleaner.clean_store('C1RCLE FRESH MART')}")
    print(f"  T0K0 SINAR → {cleaner.clean_store('T0K0 SINAR')}")
    
    print("\nAddress tests:")
    print(f"  N0. 123 J1. Raya → {cleaner.clean_address('N0. 123 J1. Raya')}")
    print(f"  Te1p: 08123456789 → {cleaner.clean_address('Te1p: 08123456789')}")
    
    print("\nNumber tests:")
    print(f"  1O.3O → {cleaner.clean_number('1O.3O')}")
    print(f"  25.8S → {cleaner.clean_number('25.8S')}")
