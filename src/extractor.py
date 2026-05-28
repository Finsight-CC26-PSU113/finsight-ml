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
        """Extract nama toko. Gabungkan beberapa baris STORE untuk nama lengkap.
        
        Filter:
        - Skip baris yang murni angka/numerik (misclassified)
        - Skip baris yang terlalu pendek (<3 alpha chars)
        - Prioritize baris di paling atas (y_min terkecil)
        """
        import re
        
        if not store_lines:
            return ""
        
        # Filter out invalid store name candidates
        valid_lines = []
        for line in store_lines:
            text = line['text'].strip()
            
            # Skip empty or too short
            if not text or len(text) < 3:
                continue
            
            # Skip lines that are mostly numbers (price/total misclassified as store)
            alpha_count = sum(c.isalpha() for c in text)
            digit_count = sum(c.isdigit() for c in text)
            if alpha_count < 3:
                continue
            if digit_count > alpha_count:
                continue
            
            # Skip lines that look like dates
            if re.search(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', text):
                continue
            
            # Skip pure currency/number patterns
            if re.match(r'^[\d\s\.,\*\-]+$', text):
                continue
            
            # Skip lines that look like phone/contact
            if re.match(r'^(tel|phone|fax|hp)[\s:.]', text, re.IGNORECASE):
                continue
            
            valid_lines.append(line)
        
        if not valid_lines:
            return ""
        
        # Sort by y_min (top first) — store name is usually at top
        sorted_lines = sorted(valid_lines, key=lambda l: l.get('y_min', 0))
        
        # Take first 3 lines for full store name (handles multi-line names)
        store_parts = []
        for line in sorted_lines[:3]:
            raw_text = line['text'].strip()
            cleaned_text = self.cleaner.clean_store(raw_text)
            if cleaned_text and len(cleaned_text) > 2:
                store_parts.append(cleaned_text)
        
        if not store_parts:
            return ""
        
        # Join with space, normalize to Title Case
        full_store_name = " ".join(store_parts)
        full_store_name = full_store_name.title()
        
        return full_store_name
    
    def _extract_date(self, date_lines: list[dict]) -> str:
        """Extract tanggal. Support format Indonesia, Malaysia, English, dan word months.
        
        Supported formats:
        - DD/MM/YYYY, DD-MM-YY, DD.MM.YY (numeric)
        - YYYY-MM-DD (ISO)
        - "Aug 19, 2024", "Mar 14 2025" (English word month)
        - "19 Agu 2024", "14 Maret 2025" (Indonesian word month)
        - With time: "Aug 19, 2024 6:32:54 PM", "DD/MM/YYYY HH:MM"
        """
        import re

        # Numeric date patterns
        numeric_date_patterns = [
            r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',           # YYYY-MM-DD (first to avoid partial)
            r'(\d{1,2}[/\-\.]\s*\d{1,2}[/\-\.]\s*\d{2,4})',  # DD/MM/YYYY (with spaces)
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',         # DD/MM/YYYY, DD-MM-YY
        ]
        
        # Word-month patterns (English & Indonesian)
        word_month_patterns = [
            # "Aug 19, 2024" or "Mar 14 2025"
            (r'((jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}[\s,]*\d{2,4})', 'en_word_first'),
            # "19 Aug 2024" or "14 Mar 2025"
            (r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4})', 'en_word_middle'),
            # Indonesian: "19 Agustus 2024", "14 Maret 2025"
            (r'(\d{1,2}\s+(jan(uari)?|feb(ruari)?|mar(et)?|apr(il)?|mei|jun(i)?|jul(i)?|agu(stus)?|sep(tember)?|okt(ober)?|nov(ember)?|des(ember)?)\s+\d{2,4})', 'id_word'),
        ]
        
        # Time patterns (only valid AS DATE if accompanied by date — standalone time is rejected)
        time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)'

        def find_date_in_text(text: str) -> str:
            """Find first match of any date pattern."""
            # Try word-month first (more specific)
            for pattern, _ in word_month_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    date_str = match.group(1).strip()
                    # Try to also capture time if it follows
                    full_pattern = re.escape(date_str) + r'\s*' + time_pattern
                    full_match = re.search(full_pattern, text, re.IGNORECASE)
                    if full_match:
                        return f"{date_str} {full_match.group(1)}".strip()
                    return date_str
            
            # Then numeric
            for pattern in numeric_date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    date_str = match.group(1).strip()
                    date_str = re.sub(r'\s+', '', date_str)
                    # Try to capture time
                    time_match = re.search(time_pattern, text, re.IGNORECASE)
                    if time_match:
                        return f"{date_str} {time_match.group(1)}".strip()
                    return date_str
            return ""

        # Primary: use labeled DATE lines
        if date_lines:
            # Sort by y_min — prefer earlier date (transaction date is usually before "Printed" footer)
            sorted_lines = sorted(date_lines, key=lambda l: l.get('y_min', 0))
            
            # Try each line individually first (early lines = transaction date)
            for line in sorted_lines:
                line_text = line['text'].strip()
                # Skip lines that are clearly time-only (no date pattern)
                if re.match(r'^\s*\d{1,2}:\d{2}', line_text) and not re.search(r'\d{4}|\d{1,2}[/\-\.]', line_text):
                    continue
                # Skip lines starting with "Printed" or "Cetak" — that's footer date
                if re.match(r'^\s*(printed|cetak|dicetak)', line_text, re.IGNORECASE):
                    continue
                result = find_date_in_text(line_text)
                if result:
                    return result
            
            # Fallback: combine all DATE-labeled lines and search
            all_date_text = " ".join([line['text'].strip() for line in sorted_lines])
            result = find_date_in_text(all_date_text)
            if result:
                return result
            
            # Last resort: return first non-empty raw text
            for line in sorted_lines:
                if line['text'].strip():
                    return line['text'].strip()
        
        return ""
    
    def _extract_items(self, item_lines: list[dict]) -> list[dict]:
        """Extract list item belanja dengan merging baris yang berdekatan.
        
        Strategi:
        1. Filter zone — items biasanya di tengah struk (15%-75% dari atas)
        2. Blacklist keywords yang bukan item
        3. Filter pure number lines (price tanpa nama)
        4. Merge multi-line item names
        """
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
            'goods', 'barang', 'product', 'produk',
            # Misc receipt keywords
            'reg', 'register', 'void', 'cancel', 'refund', 'return',
            'open', 'close', 'shift', 'balance', 'saldo',
            # Address/contact
            'jl.', 'jalan', 'jl ', 'tel:', 'telp', 'phone', 'fax', 'email',
            'www.', '.com', 'website',
            # Date/time keywords
            'date', 'tanggal', 'tgl', 'time', 'jam', 'tarikh',
            # Store-related
            'sdn bhd', 'sdn. bhd', 'pt.', 'cv.',
            # Common footer text
            'http', 'follow', 'instagram', 'facebook', 'whatsapp',
        ]
        
        # Sort by y_min (top to bottom)
        sorted_lines = sorted(item_lines, key=lambda l: l.get('y_min', 0))
        
        # Filter zone: items biasanya di tengah (15% - 75%)
        # Use y_min to detect items not at very top or very bottom
        filtered = []
        for line in sorted_lines:
            y = line.get('y_min', 0)
            text = line['text'].strip()
            text_lower = text.lower()
            
            # Skip if at top 10% (likely store/header) or bottom 10% (likely footer/total)
            if y < 0.10 or y > 0.92:
                continue
            
            # Skip very short
            if len(text) < 3:
                continue
            
            # Skip blacklisted
            if any(keyword in text_lower for keyword in blacklist_keywords):
                continue
            
            # Skip pure numeric (no item name)
            if re.match(r'^[\d\s\.,\*\-\/x@xX]+$', text):
                continue
            
            # Skip pure currency markers
            if text.upper() in ['RM', 'RP', 'IDR', 'SR', '$', 'USD']:
                continue
            
            # Must have at least 3 alpha chars (real item name has letters)
            alpha_count = sum(c.isalpha() for c in text)
            if alpha_count < 3:
                continue
            
            filtered.append(line)
        
        if not filtered:
            return []
        
        # Group lines that are close together (same item split across lines)
        merged_lines = []
        current_group = []
        last_y = None
        
        for line in filtered:
            y_pos = line.get('y_min', 0)
            
            # Within same row → group together
            if last_y is not None and abs(y_pos - last_y) < 0.025:  # 2.5% of image height
                current_group.append(line)
            else:
                if current_group:
                    merged_lines.append(current_group)
                current_group = [line]
            
            last_y = y_pos
        
        if current_group:
            merged_lines.append(current_group)
        
        # Parse each merged group as one item
        items = []
        for group in merged_lines:
            # Sort by x_min (left to right) for correct reading order
            group_sorted = sorted(group, key=lambda l: l.get('x_min', 0))
            combined_text = " ".join([l['text'].strip() for l in group_sorted])
            
            item = self._parse_item_line(combined_text)
            if item and item['name'] and len(item['name']) >= 3:
                # Final check: name must have alphabetic content
                if sum(c.isalpha() for c in item['name']) >= 3:
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
        
        Strategi prioritas (tertinggi → terendah):
        1. Eksplisit "grand total", "total bayar", "total akhir" → grand_total
        2. "total" generic (tanpa "sub") → grand_total kandidat
        3. "subtotal" / "sub total" → subtotal (BUKAN grand total)
        4. Fallback: subtotal + tax - discount, atau cash - change
        
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
        
        # Sort by y_min so later lines (typically grand total) win on tie
        sorted_total_lines = sorted(total_lines, key=lambda l: l.get('y_min', 0))
        
        for line in sorted_total_lines:
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
            
            # IMPORTANT: Check 'subtotal' BEFORE 'total' (since "subtotal" contains "total")
            is_subtotal = any(k in text_lower for k in ['subtotal', 'sub total', 'sub-total', 'jumlah'])
            is_explicit_grand = any(k in text_lower for k in [
                'grand total', 'total bayar', 'total amount', 'total pembayaran',
                'total akhir', 'total belanja', 'total tagihan', 'nett total', 'net total'
            ])
            # Generic "total" (only if NOT subtotal AND NOT a sub-charge like service charge)
            # Also exclude lines that have "service" or "charge" — these are sub-fees, not the main total
            is_service_charge = any(k in text_lower for k in [
                'service charge', 'service', 'charge', 'biaya layanan', 'biaya'
            ])
            is_tax_line = any(k in text_lower for k in [
                'tax', 'pajak', 'ppn', 'gst', 'vat', 'pb1'
            ])
            is_discount_line = any(k in text_lower for k in [
                'discount', 'diskon', 'potongan', 'disc', 'voucher', 'promo'
            ])
            is_cash_line = any(k in text_lower for k in ['cash', 'tunai', 'bayar', 'paid'])
            is_change_line = any(k in text_lower for k in ['change', 'kembali', 'kembalian'])
            
            # Generic "total" wins ONLY if not in any other category
            has_total_kw = (
                'total' in text_lower 
                and not is_subtotal 
                and not is_service_charge 
                and not is_tax_line
                and not is_discount_line
                and not is_cash_line
                and not is_change_line
            )
            
            for num_str in numbers:
                val = self._parse_number(num_str)
                if val <= 0 or val > 100000000:  # Skip invalid or unrealistic values
                    continue
                
                # PRIORITAS 1: Explicit Grand Total
                if is_explicit_grand:
                    grand_total_candidates.append((val, 10))
                # PRIORITAS 2: Subtotal (cek SEBELUM total generic)
                elif is_subtotal:
                    subtotal_candidates.append(val)
                # PRIORITAS 3: Service charge → its own bucket (NOT grand total)
                elif is_service_charge:
                    pass  # tracked but not as grand_total candidate
                # PRIORITAS 4: Tax line
                elif is_tax_line:
                    tax_candidates.append(val)
                # PRIORITAS 5: Discount
                elif is_discount_line:
                    discount_candidates.append(val)
                # PRIORITAS 6: Cash/payment
                elif is_cash_line:
                    cash_candidates.append(val)
                # PRIORITAS 7: Change
                elif is_change_line:
                    change_candidates.append(val)
                # PRIORITAS 8: Generic "total"
                elif has_total_kw:
                    grand_total_candidates.append((val, 5))
                elif line.get('is_total_target'):
                    # From cascading logic
                    grand_total_candidates.append((val, 3))
        
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
            # Sort by priority (desc), then by value (desc)
            grand_total_candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
            totals['grand_total'] = grand_total_candidates[0][0]
        
        # Fallback: Calculate grand total if not found
        if totals['grand_total'] == 0.0:
            if totals['subtotal'] > 0 and totals['tax'] > 0:
                calculated_total = totals['subtotal'] + totals['tax'] - totals['discount']
                if calculated_total > 0:
                    totals['grand_total'] = calculated_total
            elif totals['subtotal'] > 0:
                calculated_total = totals['subtotal'] - totals['discount']
                if calculated_total > 0:
                    totals['grand_total'] = calculated_total
            elif totals['cash'] > 0 and totals['change'] > 0:
                totals['grand_total'] = totals['cash'] - totals['change']
        
        # Final sanity check: grand_total should be >= subtotal
        if totals['grand_total'] > 0 and totals['subtotal'] > 0:
            if totals['grand_total'] < totals['subtotal']:
                # Misclassified — use subtotal+tax as grand total
                totals['grand_total'] = totals['subtotal'] + totals['tax'] - totals['discount']
        
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
