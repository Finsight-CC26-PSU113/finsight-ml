"""
OCR FinSight - Receipt Data Extractor
Regex-based value extraction dari classified OCR lines.
"""

import re
from typing import Optional
from src.text_cleaner import OCRTextCleaner


class ReceiptExtractor:
    """Extract structured data dari baris OCR yang sudah di-classify."""
    
    def __init__(self):
        self.cleaner = OCRTextCleaner()
    
    def extract(self, classified_lines: list[dict]) -> dict:
        """Extract semua field dari classified lines.
        
        Args:
            classified_lines: List of dicts dengan key 'text' dan 'predicted_class'
            
        Returns:
            Dict berisi: store, date, items, total, raw_lines
        """
        # Perform context-aware correction & cascading on classified lines
        total_keywords = ['total', 'subtota', 'amount', 'rounding', 'tunai', 'kembali', 'change', 'cash', 'bayar']
        for i in range(len(classified_lines)):
            text_lower = classified_lines[i]['text'].lower()
            # 1. Safeguard: Jika teks secara eksplisit mengandung kata kunci total, pastikan berlabel TOTAL_PAYMENT
            if any(k in text_lower for k in total_keywords):
                classified_lines[i]['predicted_class'] = 'TOTAL_PAYMENT'
            
            # 2. Cascading: Jika baris saat ini TOTAL_PAYMENT, intip hingga 2 baris ke bawah
            is_explicit_total = any(k in text_lower for k in ['total', 'subtota', 'amount', 'bayar'])
            if classified_lines[i].get('predicted_class') == 'TOTAL_PAYMENT':
                for offset in range(1, 3):
                    if i + offset < len(classified_lines):
                        target_line = classified_lines[i+offset]
                        target_text = target_line['text'].strip()
                        if any(c.isdigit() for c in target_text) and len(target_text) <= 15:
                            alphas = sum(c.isalpha() for c in target_text)
                            if alphas <= 5:
                                target_line['predicted_class'] = 'TOTAL_PAYMENT'
                                if is_explicit_total:
                                    target_line['is_total_target'] = True
                        
        # Group lines by class
        grouped = {}
        for line in classified_lines:
            cls = line.get('predicted_class', 'OTHER')
            grouped.setdefault(cls, []).append(line)
        
        result = {
            'store': self._extract_store(grouped.get('STORE', [])),
            'date': self._extract_date(grouped.get('DATE', [])),
            'items': self._extract_items(grouped.get('ITEM_DESC', []) + grouped.get('ITEM_PRICE/QTY', [])),
            'totals': self._extract_total(grouped.get('TOTAL_PAYMENT', [])),
            'total': 0.0,  # Will be set from totals['grand_total']
            'address': self._extract_address(grouped.get('ADDRESS_CONTACT', [])),
            'raw_lines': [
                {
                    'text': l['text'], 
                    'class': l.get('predicted_class', 'OTHER'),
                    'confidence': float(l.get('class_confidence', 0.0)),
                    'bbox': [[float(pt[0]), float(pt[1])] for pt in l['bbox']] if 'bbox' in l else None,
                    'x_min': float(l.get('x_min', 0)),
                    'y_min': float(l.get('y_min', 0)),
                    'width': float(l.get('width', 0)),
                    'height': float(l.get('height', 0))
                }
                for l in classified_lines
            ],
        }
        
        # Set total from grand_total for backward compatibility
        result['total'] = result['totals']['grand_total']
        
        return result
    
    def _extract_store(self, store_lines: list[dict]) -> str:
        """Extract nama toko. Gabungkan beberapa baris STORE untuk nama lengkap."""
        if not store_lines:
            return ""
        
        # Sort by y_min (top to bottom) and confidence
        sorted_lines = sorted(store_lines, key=lambda l: (l.get('y_min', 0), -l.get('class_confidence', 0)))
        
        # Ambil maksimal 3 baris pertama untuk nama toko lengkap
        store_parts = []
        for line in sorted_lines[:3]:
            raw_text = line['text'].strip()
            if raw_text and len(raw_text) > 2:  # Skip very short text
                cleaned_text = self.cleaner.clean_store(raw_text)
                store_parts.append(cleaned_text)
        
        # Gabungkan dengan spasi
        full_store_name = " ".join(store_parts)
        
        # Normalize case: Title Case untuk nama toko
        full_store_name = full_store_name.title()
        
        return full_store_name
    
    def _extract_date(self, date_lines: list[dict]) -> str:
        """Extract tanggal. Gabungkan beberapa baris DATE untuk tanggal lengkap."""
        if not date_lines:
            return ""
        
        # Gabungkan semua baris DATE untuk mendapatkan tanggal lengkap
        all_date_text = " ".join([line['text'].strip() for line in date_lines])
        
        date_patterns = [
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',  # DD/MM/YYYY or DD-MM-YYYY
            r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',      # YYYY-MM-DD
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4})',  # DD Month YYYY
            r'(\d{1,2}[/\-\.]\s*\d{1,2}[/\-\.]\s*\d{2,4})',  # DD/ MM/ YYYY (with spaces)
        ]
        
        # Try to find date pattern in combined text
        for pattern in date_patterns:
            match = re.search(pattern, all_date_text, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                # Clean up spaces in date
                date_str = re.sub(r'\s+', '', date_str)
                return date_str
        
        # Fallback: return combined text (cleaned)
        return all_date_text.strip()
    
    def _extract_items(self, item_lines: list[dict]) -> list[dict]:
        """Extract list item belanja dengan merging baris yang berdekatan."""
        if not item_lines:
            return []
        
        # Comprehensive blacklist keywords
        blacklist_keywords = [
            # Receipt metadata
            'item:', 'qty:', 'quantity:', 'price:', 'amount:', 'total:', 'subtotal:',
            # Staff & service
            'cashier', 'kasir', 'waiter', 'waitress', 'server', 'staff', 'operator',
            # Customer info
            'customer', 'pelanggan', 'member', 'membership', 'card no', 'card number',
            # Table & location
            'table', 'meja', 'pax', 'guest', 'tamu', 'room', 'kamar',
            # Transaction info
            'receipt', 'struk', 'bill', 'invoice', 'transaction', 'transaksi', 'trx',
            'order', 'pesanan', 'no.', 'ref', 'reference',
            # Tender & payment
            'tender', 'payment', 'pembayaran', 'cash', 'tunai', 'card', 'credit',
            'debit', 'change', 'kembali', 'kembalian',
            # Discounts & charges
            'discount', 'diskon', 'potongan', 'promo', 'voucher', 'coupon',
            'service charge', 'tax', 'pajak', 'gst', 'vat', 'ppn',
            # Totals
            'total', 'subtotal', 'grand total', 'amount', 'jumlah',
            # Loyalty & points
            'point', 'points', 'reward', 'saving', 'hemat', 'earned',
            # Greetings & messages
            'thank', 'terima', 'kasih', 'welcome', 'selamat', 'datang',
            'please', 'silakan', 'come again', 'visit',
            # Categories & headers
            'goods', 'barang', 'item', 'product', 'produk',
            'food', 'makanan', 'drink', 'minuman', 'beverage',
            # Misc
            'reg', 'register', 'void', 'cancel', 'refund', 'return',
            'open', 'close', 'shift', 'balance', 'saldo',
        ]
        
        # Sort by y_min (top to bottom)
        sorted_lines = sorted(item_lines, key=lambda l: l.get('y_min', 0))
        
        # Group lines that are close together (same item split across lines)
        merged_lines = []
        current_group = []
        last_y = None
        
        for line in sorted_lines:
            text = line['text'].strip()
            text_lower = text.lower()
            y_pos = line.get('y_min', 0)
            
            # Skip very short or blacklisted
            if len(text) < 2:
                continue
            if any(keyword in text_lower for keyword in blacklist_keywords):
                continue
            
            # Check if this line is close to previous (same item)
            if last_y is not None and abs(y_pos - last_y) < 30:  # Within 30 pixels
                current_group.append(line)
            else:
                # Save previous group
                if current_group:
                    merged_lines.append(current_group)
                # Start new group
                current_group = [line]
            
            last_y = y_pos
        
        # Don't forget last group
        if current_group:
            merged_lines.append(current_group)
        
        # Parse each merged group as one item
        items = []
        for group in merged_lines:
            # Combine text from group
            combined_text = " ".join([l['text'].strip() for l in group])
            
            # Skip if just numbers
            if re.match(r'^[\d\s\.,\*\-]+$', combined_text):
                continue
            
            item = self._parse_item_line(combined_text)
            if item and item['name'] and len(item['name']) > 2:
                items.append(item)
        
        return items
    
    def _parse_item_line(self, text: str) -> Optional[dict]:
        """Parse satu baris item menjadi {name, qty, price} dengan logic yang lebih baik."""
        if not text or len(text) < 3:
            return None
        
        # Clean item name first
        text = self.cleaner.clean_item(text)
        
        item = {'name': '', 'qty': 1, 'price': 0.0, 'raw': text}
        
        # Pattern 1: "Item Name  2 x 5000" atau "Item Name  2*5000"
        qty_price_match = re.search(r'(\d+)\s*[xX\*]\s*([\d.,]+)', text)
        if qty_price_match:
            item['qty'] = int(qty_price_match.group(1))
            item['price'] = self._parse_number(qty_price_match.group(2))
            item['name'] = text[:qty_price_match.start()].strip()
            return item if item['name'] else None
        
        # Pattern 2: "1.40*1" atau "9.80 SR" atau "@9.80" (price with special format)
        special_price_patterns = [
            (r'([\d.,]+)\s*\*\s*(\d+)', True),  # 1.40*1 (has qty)
            (r'([\d.,]+)\s+(?:SR|RM|IDR|Rp|rp)', False),  # 9.80 SR
            (r'@\s*([\d.,]+)', False),  # @9.80
        ]
        
        for pattern, has_qty in special_price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                price_str = match.group(1)
                price = self._parse_number(price_str)
                if price > 0:
                    item['price'] = price
                    # Try to extract qty if pattern has it
                    if has_qty and len(match.groups()) > 1:
                        try:
                            item['qty'] = int(match.group(2))
                        except (ValueError, IndexError):
                            pass
                    item['name'] = text[:match.start()].strip()
                    if not item['name']:
                        # Name might be after the price
                        item['name'] = text[match.end():].strip()
                    return item if item['name'] else None
        
        # Pattern 3: "Item Name    10000" atau "Item Name    10.000" (price at end)
        # More aggressive: look for any number at the end that could be a price
        price_match = re.search(r'\s+([\d.,]{3,})\s*$', text)
        if price_match:
            price = self._parse_number(price_match.group(1))
            # Lower threshold for price detection
            if price >= 50 and price < 10000000:
                item['price'] = price
                item['name'] = text[:price_match.start()].strip()
                return item if item['name'] else None
        
        # Pattern 4: Price in the middle "Item 5000 Name" (less common)
        middle_price_match = re.search(r'\s([\d.,]{4,})\s', text)
        if middle_price_match:
            price = self._parse_number(middle_price_match.group(1))
            if 50 <= price < 10000000:
                item['price'] = price
                # Name is before and after price
                before = text[:middle_price_match.start()].strip()
                after = text[middle_price_match.end():].strip()
                item['name'] = f"{before} {after}".strip()
                return item if item['name'] else None
        
        # Pattern 5: Look for price anywhere with currency symbol
        currency_match = re.search(r'(?:Rp|RM|IDR|rp)\s*([\d.,]+)', text, re.IGNORECASE)
        if currency_match:
            price = self._parse_number(currency_match.group(1))
            if price >= 50:
                item['price'] = price
                # Name is everything except the price part
                item['name'] = text[:currency_match.start()].strip() + " " + text[currency_match.end():].strip()
                item['name'] = item['name'].strip()
                return item if item['name'] else None
        
        # Pattern 6: Just item name without clear price
        # Check if text has mostly letters (item name)
        alpha_count = sum(c.isalpha() for c in text)
        digit_count = sum(c.isdigit() for c in text)
        
        if alpha_count > digit_count and alpha_count >= 3:
            item['name'] = text
            return item
        
        return None
    
    def _extract_total(self, total_lines: list[dict]) -> dict:
        """Extract multiple totals dengan prioritas GRAND TOTAL (setelah pajak).
        
        Returns:
            Dict dengan keys: grand_total, subtotal, discount, tax, cash, change
        """
        totals = {
            'grand_total': 0.0,
            'subtotal': 0.0,
            'discount': 0.0,
            'tax': 0.0,
            'cash': 0.0,
            'change': 0.0
        }
        
        # Kategorisasi berdasarkan keyword dengan prioritas
        grand_total_candidates = []
        subtotal_candidates = []
        discount_candidates = []
        tax_candidates = []
        cash_candidates = []
        change_candidates = []
        
        for line in total_lines:
            text = line['text']
            text_lower = text.lower()
            
            # Safeguard: Skip NPWP, phone numbers, transaction IDs
            if any(k in text_lower for k in ['npwp', 'tel', 'fax', 'phone', 'call', 'roc', 'gst no', 'trxid', 'member']):
                continue
            
            # Skip complex ID patterns
            digits_count = len(re.findall(r'\d', text))
            if digits_count >= 10 and (text.count('-') >= 1 or text.count('.') >= 2):
                continue
            
            # Clean OCR typos
            text_clean = text.replace('O', '0').replace('o', '0')
            numbers = re.findall(r'[\d]+[.,]?[\d]*', text_clean)
            
            for num_str in numbers:
                val = self._parse_number(num_str)
                if val <= 0 or val > 100000000:  # Skip invalid or unrealistic values
                    continue
                
                # PRIORITAS 1: Grand Total (setelah pajak) - HIGHEST PRIORITY
                if any(k in text_lower for k in ['grand total', 'total bayar', 'total amount', 'total pembayaran', 'total akhir']):
                    grand_total_candidates.append((val, 10))  # Priority 10
                # PRIORITAS 2: Total (generic) - bisa jadi grand total
                elif 'total' in text_lower and 'sub' not in text_lower:
                    grand_total_candidates.append((val, 5))  # Priority 5
                # PRIORITAS 3: Subtotal (sebelum pajak) - LOWER PRIORITY
                elif any(k in text_lower for k in ['subtotal', 'sub total', 'sub-total', 'jumlah']):
                    subtotal_candidates.append(val)
                # Other categories
                elif any(k in text_lower for k in ['discount', 'diskon', 'potongan', 'disc']):
                    discount_candidates.append(val)
                elif any(k in text_lower for k in ['tax', 'pajak', 'ppn', 'gst', 'vat']):
                    tax_candidates.append(val)
                elif any(k in text_lower for k in ['cash', 'tunai', 'bayar', 'paid']):
                    cash_candidates.append(val)
                elif any(k in text_lower for k in ['change', 'kembali', 'kembalian']):
                    change_candidates.append(val)
                elif line.get('is_total_target'):
                    # From cascading logic
                    grand_total_candidates.append((val, 3))  # Priority 3
        
        # Assign values
        if subtotal_candidates:
            totals['subtotal'] = max(subtotal_candidates)
        if discount_candidates:
            totals['discount'] = max(discount_candidates)
        if tax_candidates:
            totals['tax'] = max(tax_candidates)
        if cash_candidates:
            totals['cash'] = max(cash_candidates)
        if change_candidates:
            totals['change'] = max(change_candidates)
        
        # Grand Total Logic: Prioritize highest priority candidate
        if grand_total_candidates:
            # Sort by priority (descending), then by value (descending)
            grand_total_candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
            totals['grand_total'] = grand_total_candidates[0][0]
        
        # Fallback: Calculate grand total if not found
        if totals['grand_total'] == 0.0:
            if totals['subtotal'] > 0 and totals['tax'] > 0:
                # grand_total = subtotal + tax - discount
                calculated_total = totals['subtotal'] + totals['tax'] - totals['discount']
                if calculated_total > 0:
                    totals['grand_total'] = calculated_total
            elif totals['subtotal'] > 0:
                # If no tax, grand_total = subtotal - discount
                calculated_total = totals['subtotal'] - totals['discount']
                if calculated_total > 0:
                    totals['grand_total'] = calculated_total
            elif totals['cash'] > 0 and totals['change'] > 0:
                # grand_total = cash - change
                totals['grand_total'] = totals['cash'] - totals['change']
        
        # Final check: If grand_total < subtotal, use subtotal + tax
        if totals['grand_total'] > 0 and totals['subtotal'] > 0:
            if totals['grand_total'] < totals['subtotal']:
                # Grand total should be >= subtotal
                totals['grand_total'] = totals['subtotal'] + totals['tax']
        
        return totals
    
    def _extract_address(self, address_lines: list[dict]) -> str:
        """Extract alamat. Gabungkan semua baris ADDRESS."""
        if not address_lines:
            return ""
        # Clean each address line
        texts = [self.cleaner.clean_address(l['text'].strip()) for l in address_lines]
        return ", ".join(texts)
    
    def _parse_number(self, num_str: str) -> float:
        """Parse string angka (handle format Indonesia: 10.000 dan 10,00)."""
        num_str = num_str.strip()
        if not num_str:
            return 0.0
        
        # Jika ada titik dan koma: "10.000,00" → 10000.00
        if '.' in num_str and ',' in num_str:
            num_str = num_str.replace('.', '').replace(',', '.')
        # Jika hanya koma: "10,00" → 10.00
        elif ',' in num_str:
            parts = num_str.split(',')
            if len(parts[-1]) == 2:  # decimal
                num_str = num_str.replace(',', '.')
            else:  # thousands separator
                num_str = num_str.replace(',', '')
        # Jika hanya titik: "10.000" (thousands) vs "10.00" (decimal)
        elif '.' in num_str:
            parts = num_str.split('.')
            if len(parts[-1]) == 3:  # thousands separator
                num_str = num_str.replace('.', '')
        
        try:
            return float(num_str)
        except ValueError:
            return 0.0
