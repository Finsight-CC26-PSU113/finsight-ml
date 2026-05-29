"""
OCR FinSight - Receipt Data Extractor
Regex-based value extraction dari classified OCR lines.
"""

import re
import numpy as np
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
        # NOTE: 'cash' diganti dengan boundary regex agar tidak match "cashier",
        # 'change' juga di-boundary agar tidak match kata lain yang kebetulan mengandungnya.
        total_substrings = ['total', 'subtota', 'amount', 'rounding', 'tunai', 'kembali', 'bayar']
        total_word_re = re.compile(r'\b(cash|change)\b', re.IGNORECASE)
        # Eksklusi: kata-kata yang TIDAK boleh dianggap TOTAL_PAYMENT meski mengandung substring "tot..."
        total_exclusions_re = re.compile(r'\b(cashier|kasir|server|waiter|operator)\b', re.IGNORECASE)

        for i in range(len(classified_lines)):
            text = classified_lines[i]['text']
            text_lower = text.lower()

            # Skip baris yang jelas-jelas BUKAN total (cashier name dll.)
            if total_exclusions_re.search(text):
                continue

            # 1. Safeguard: paksa TOTAL_PAYMENT bila teks mengandung kata kunci total
            has_substring = any(k in text_lower for k in total_substrings)
            has_word = bool(total_word_re.search(text))
            if has_substring or has_word:
                classified_lines[i]['predicted_class'] = 'TOTAL_PAYMENT'

            # 2. Cascading DINONAKTIFKAN.
            # Alasan: cascade mark baris angka di bawah "Subtota]" sebagai total_target,
            # tapi baris berikutnya bisa saja service charge value (223,000) yang bukan grand total.
            # Sekarang kita andalkan keyword matching per-line di _extract_total saja.
                        
        # Group lines by class
        grouped = {}
        for line in classified_lines:
            cls = line.get('predicted_class', 'OTHER')
            grouped.setdefault(cls, []).append(line)
        
        result = {
            'store': self._extract_store(grouped.get('STORE', []), classified_lines),
            'date': self._extract_date(grouped.get('DATE', []), classified_lines),
            'items': self._extract_items(classified_lines),
            'totals': self._extract_total(classified_lines),
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
    
    def _extract_store(self, store_lines: list[dict], all_lines: list[dict] | None = None) -> str:
        """Extract nama toko. Gabungkan beberapa baris STORE untuk nama lengkap.
        
        Filter:
        - Skip baris yang murni angka/numerik (misclassified)
        - Skip baris yang terlalu pendek (<3 alpha chars)
        - Prioritize baris di paling atas (y_min terkecil)
        - Fallback: bila STORE kosong, ambil baris paling atas dari ADDRESS_CONTACT
          (classifier kadang salah label STORE sebagai ADDRESS_CONTACT pada struk).
        """
        import re

        # Bila tidak ada STORE label, fallback ke baris top-most dari ADDRESS_CONTACT/OTHER
        candidates = list(store_lines) if store_lines else []
        if not candidates and all_lines:
            for line in all_lines:
                if line.get('predicted_class') in ('ADDRESS_CONTACT', 'OTHER'):
                    candidates.append(line)
        
        if not candidates:
            return ""
        
        # Filter out invalid store name candidates
        valid_lines = []
        for line in candidates:
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
    
    def _extract_date(self, date_lines: list[dict], all_lines: list[dict] | None = None) -> str:
        """Extract tanggal. Support format Indonesia, Malaysia, English, dan word months.
        
        Supported formats:
        - DD/MM/YYYY, DD-MM-YY, DD.MM.YY (numeric, dengan validasi day/month range)
        - YYYY-MM-DD (ISO)
        - "Aug 19, 2024", "Mar 14 2025" (English word month)
        - "19 Agu 2024", "14 Maret 2025" (Indonesian word month)
        - With time: "Aug 19, 2024 6:32:54 PM", "DD/MM/YYYY HH:MM"
        """
        import re

        # Numeric date patterns (separator harus konsisten supaya "6.32.54" ditolak via validasi)
        numeric_date_patterns = [
            (r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', 'iso'),     # YYYY-MM-DD
            (r'(\d{1,2})[/\-\.]\s*(\d{1,2})[/\-\.]\s*(\d{2,4})', 'dmy'),  # DD/MM/YYYY (with spaces)
        ]
        
        # Word-month patterns (English & Indonesian)
        # NOTE: year wajib 4 digit (\d{4}) untuk hindari false positive
        word_month_patterns = [
            # "Aug 19, 2024" or "Mar 14 2025" (with optional spaces)
            (r'((jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}[\s,]*\d{4})', 'en_word_first'),
            # "Aug192024" (no spaces, OCR sometimes drops them)
            (r'((jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*\d{1,2}\s*[,.]?\s*\d{4})', 'en_word_first_compact'),
            # "19 Aug 2024" or "14 Mar 2025"
            (r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4})', 'en_word_middle'),
            # "03-Feb-2023" or "14.Mar.2025" (dash/dot separator)
            (r'(\d{1,2}[\-\.](jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\-\.]\d{4})', 'en_word_dash'),
            # Indonesian: "19 Agustus 2024", "14 Maret 2025"
            (r'(\d{1,2}\s+(jan(uari)?|feb(ruari)?|mar(et)?|apr(il)?|mei|jun(i)?|jul(i)?|agu(stus)?|sep(tember)?|okt(ober)?|nov(ember)?|des(ember)?)\s+\d{4})', 'id_word'),
        ]
        
        # Time patterns (only valid AS DATE if accompanied by date — standalone time is rejected)
        time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)'

        def is_valid_dmy(d: int, m: int, y: int) -> bool:
            return 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= (y if y >= 100 else 2000 + y) <= 2100

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
            
            # Then numeric (with validation)
            for pattern, kind in numeric_date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if kind == 'iso':
                        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    else:  # dmy
                        d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    if not is_valid_dmy(d, m, y):
                        continue  # invalid like "6.32.54", reject
                    date_str = match.group(0).strip()
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

        # Secondary fallback: scan SEMUA baris untuk pattern tanggal valid
        # Berguna ketika classifier salah label STORE/ADDRESS sebagai DATE atau sebaliknya.
        # PENTING: hanya gabungkan baris bila y_min mereka berdekatan, agar tidak salah merge
        # baris dari area yang berbeda (mis. "1" dari Cnt + "Aug 19," dari date area).
        if all_lines:
            sorted_all = sorted(all_lines, key=lambda l: l.get('y_min', 0))
            Y_TOLERANCE = 0.03  # ~5% of image height; hanya gabungkan baris yang benar-benar berdekatan
            for i in range(len(sorted_all)):
                for window in (1, 2, 3):
                    end = i + window
                    if end > len(sorted_all):
                        continue
                    chunk_lines = sorted_all[i:end]
                    # Pastikan semua baris di window benar-benar berdekatan secara vertikal
                    y_values = [l.get('y_min', 0) for l in chunk_lines]
                    if max(y_values) - min(y_values) > Y_TOLERANCE:
                        continue
                    chunk = " ".join(l['text'].strip() for l in chunk_lines)
                    if re.match(r'^\s*(printed|cetak|dicetak)', chunk, re.IGNORECASE):
                        continue
                    result = find_date_in_text(chunk)
                    if result:
                        return result

        # Last resort: return first non-empty raw text from labeled date_lines
        if date_lines:
            for line in sorted(date_lines, key=lambda l: l.get('y_min', 0)):
                if line['text'].strip():
                    return line['text'].strip()
        
        return ""
    
    def _extract_items(self, item_lines: list[dict]) -> list[dict]:
        """Extract list item belanja dengan column-aware pairing.
        
        Strategi:
        1. Filter zone — items biasanya di tengah struk (10%-92% dari atas)
        2. Pisahkan baris jadi 2 kolom: NAME (kiri, x_min < 0.5) dan PRICE (kanan, x_min >= 0.5)
        3. Untuk tiap baris harga, pair dengan baris nama yang y-nya overlap
        4. Untuk tiap baris nama, gabungkan dengan baris berdekatan vertikal di kolom kiri
        5. Skip blacklist & footer
        """
        if not item_lines:
            return []

        blacklist_keywords = [
            'item:', 'qty:', 'quantity:', 'price:', 'amount:', 'total:', 'subtotal:',
            'cashier', 'kasir', 'waiter', 'waitress', 'server', 'staff', 'operator',
            'customer', 'pelanggan', 'member', 'membership', 'card no', 'card number',
            'table', 'meja', 'pax', 'guest', 'tamu', 'room', 'kamar',
            'receipt', 'struk', 'bill', 'invoice', 'transaction', 'transaksi', 'trx',
            'order', 'pesanan', 'no.', 'ref', 'reference',
            'tender', 'payment', 'pembayaran', 'cash', 'tunai', 'card', 'credit',
            'debit', 'change', 'kembali', 'kembalian',
            'discount', 'diskon', 'potongan', 'promo', 'voucher', 'coupon',
            'service charge', 'tax', 'pajak', 'gst', 'vat', 'ppn',
            'total', 'subtotal', 'grand total', 'amount', 'jumlah',
            'point', 'points', 'reward', 'saving', 'hemat', 'earned',
            'thank', 'terima', 'kasih', 'welcome', 'selamat', 'datang',
            'please', 'silakan', 'come again', 'visit',
            'goods', 'barang', 'product', 'produk',
            'reg', 'register', 'void', 'cancel', 'refund', 'return',
            'open', 'close', 'shift', 'balance', 'saldo',
            'jl.', 'jalan', 'jl ', 'tel:', 'telp', 'phone', 'fax', 'email',
            'www.', '.com', 'website',
            'date', 'tanggal', 'tgl', 'time', 'jam', 'tarikh',
            'sdn bhd', 'sdn. bhd', 'pt.', 'cv.',
            'http', 'follow', 'instagram', 'facebook', 'whatsapp',
        ]

        def is_blacklisted(text: str) -> bool:
            text_lower = text.lower()
            return any(k in text_lower for k in blacklist_keywords)

        # Filter zone: items biasanya di tengah struk
        # Lebih ketat: 0.18 - 0.75 (di luar header & sebelum total area)
        zone_filtered = [
            l for l in item_lines
            if 0.18 <= l.get('y_min', 0) <= 0.75
            and not is_blacklisted(l['text'])
        ]
        if not zone_filtered:
            return []

        # Bagi jadi 2 kolom berdasarkan x_min:
        # - PRICE column: x_min >= 0.55 (sisi kanan struk) DAN text mengandung digit
        # - NAME column: sisanya (sisi kiri/tengah)
        price_pattern = re.compile(r'\d')
        name_lines = []
        price_lines = []
        for l in zone_filtered:
            text = l['text'].strip()
            if not text:
                continue
            x_min = l.get('x_min', 0)
            has_digit = bool(price_pattern.search(text))
            # Kolom kanan + ada digit → kandidat harga
            # TAPI skip baris yang hanya qty indicator ("PCS", "2 PCS @", "BKS @")
            if x_min >= 0.55 and has_digit:
                # Cek apakah ada angka >= 100 (harga valid) di text ini
                money_check = re.compile(r'\d{1,3}(?:[,.]\d{3})+|\d+')
                nums_in_text = money_check.findall(text.replace(' ', ''))
                has_valid_price = any(self._parse_number(n) >= 100 for n in nums_in_text)
                if has_valid_price:
                    price_lines.append(l)
                # else: skip — ini qty indicator seperti "PCS", "2 PCS @", "BKS @"
            else:
                # Skip pure number / pure punctuation di kolom kiri (artifact OCR)
                if re.match(r'^[\d\s\.,\*\-\/x@xX]+$', text):
                    continue
                if text.upper() in ['RM', 'RP', 'IDR', 'SR', '$', 'USD']:
                    continue
                if sum(c.isalpha() for c in text) < 2:
                    continue
                # Skip qty indicators yang bukan nama item
                text_upper = text.strip().upper()
                if text_upper in ['PCS', 'BTL', 'BKS', 'KG', 'GR', 'ML', 'LTR', 'DUS', 'BOX', 'SET', 'PACK', 'UNIT']:
                    continue
                # Skip "2 PCS @", "1BTL @" patterns
                if re.match(r'^\d*\s*(pcs|btl|bks|kg|gr|ml|ltr|dus|box|set|pack|unit)\s*@?\s*$', text.strip(), re.IGNORECASE):
                    continue
                name_lines.append(l)

        # ============================================================
        # PRICE-DRIVEN PAIRING (lebih robust daripada group-by-name)
        # Ide: untuk tiap baris PRICE di kolom kanan, cari baris NAME yang
        # y-nya paling overlap. Ini menghindari "monster line" yang merge
        # banyak baris OTHER yang kebetulan ada di kolom kiri.
        # ============================================================
        if not price_lines:
            return []

        # Sort price lines by y for stable iteration
        price_lines_sorted = sorted(price_lines, key=lambda l: l.get('y_min', 0))
        items = []
        used_name_lines = set()

        # Toleransi y untuk pairing: 1.0x tinggi rata-rata baris
        # Cukup untuk handle layout 2-kolom yang tidak perfectly aligned
        # tapi tetap hanya ambil 1 name TERDEKAT per price
        all_heights = [l.get('height', 0.02) for l in name_lines + price_lines]
        avg_h = (sum(all_heights) / len(all_heights)) if all_heights else 0.02
        Y_PAIR_TOL = max(avg_h * 1.0, 0.012)

        for pl in price_lines_sorted:
            p_y_center = (pl.get('y_min', 0) + pl.get('y_max', 0)) / 2

            # Parse harga dari price line
            price_text = pl['text'].strip().replace(' ', '')
            money_pattern = re.compile(r'\d{1,3}(?:[,.]\d{3})+|\d+')
            candidates = money_pattern.findall(price_text)
            price_value = 0.0
            for c in candidates:
                v = self._parse_number(c)
                if 100 <= v <= 10_000_000:
                    price_value = v
                    break
            if price_value <= 0:
                continue

            # Cari name line TERDEKAT (1 saja) di kolom kiri yang y-nya overlap
            # PENTING: name harus di ATAS atau SEJAJAR dengan price (bukan di bawah)
            # karena di struk, harga selalu sejajar/sedikit di bawah nama item
            best_name = None
            best_dist = float('inf')
            best_idx = -1
            for idx, nl in enumerate(name_lines):
                if idx in used_name_lines:
                    continue
                n_y_center = (nl.get('y_min', 0) + nl.get('y_max', 0)) / 2
                # Name harus di atas atau sejajar price (n_y_center <= p_y_center + tolerance kecil)
                if n_y_center > p_y_center + Y_PAIR_TOL * 0.3:
                    continue  # name di bawah price → skip
                dist = abs(n_y_center - p_y_center)
                if dist <= Y_PAIR_TOL and dist < best_dist:
                    best_dist = dist
                    best_name = nl
                    best_idx = idx

            if best_name is None:
                continue

            used_name_lines.add(best_idx)
            raw_name = best_name['text'].strip()
            cleaned_name = self.cleaner.clean_item(raw_name).strip()
            if not cleaned_name or sum(c.isalpha() for c in cleaned_name) < 3:
                continue

            # Skip jika name mengandung keyword total/subtotal (OCR typo "Tota]")
            name_lower = cleaned_name.lower()
            if any(k in name_lower for k in ['tota', 'subtota', 'service', 'printed', 'print']):
                continue

            items.append({
                'name': cleaned_name,
                'qty': 1,
                'price': float(price_value),
                'raw': raw_name,
            })

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

        # Pre-pass merge fragmented price tokens DINONAKTIFKAN.
        # Penyebab: pada struk dengan layout 2 kolom (label kiri + value kanan),
        # merging vertikal di kolom kanan dapat menggabung nilai dari baris yang berbeda
        # (mis. tax value + subtotal value) sehingga membentuk angka raksasa yang salah.
        # Smart split di money_pattern di bawah cukup menangani fragmen di dalam satu baris OCR.

        for line in sorted_total_lines:
            text = line['text']
            text_lower = text.lower()

            # FILTER: hanya proses baris yang mengandung keyword relevan
            # atau yang murni angka (potential value line dari total area)
            has_relevant_kw = any(k in text_lower for k in [
                'total', 'subtota', 'amount', 'service', 'charge', 'tax', 'pajak',
                'discount', 'diskon', 'cash', 'tunai', 'bayar', 'change', 'kembali',
                'rounding', 'ppn', 'gst', 'vat'
            ])
            is_pure_number = bool(re.match(r'^[\d\s,.\-]+$', text.strip()))
            if not has_relevant_kw and not is_pure_number:
                continue
            
            # Safeguard: Skip NPWP, phone numbers, transaction IDs
            if any(k in text_lower for k in ['npwp', 'tel', 'fax', 'phone', 'call', 'roc', 'gst no', 'trxid', 'member']):
                continue
            
            # Skip complex ID patterns
            digits_count = len(re.findall(r'\d', text))
            if digits_count >= 10 and (text.count('-') >= 1 or text.count('.') >= 2):
                continue
            
            # Clean OCR typos dan rapatkan whitespace dalam angka ("223 ,000" → "223,000")
            text_clean = text.replace('O', '0').replace('o', '0')
            # Hapus spasi yang berada di antara digit/koma/titik (artifact OCR)
            text_clean = re.sub(r'(\d)\s+([,.])', r'\1\2', text_clean)
            text_clean = re.sub(r'([,.])\s+(\d)', r'\1\2', text_clean)
            # Tangkap setiap "kelompok angka" valid sebagai uang.
            # Pattern (urutan penting karena alternasi greedy):
            #   1. AAA(,BBB)+    → format ribuan: "1,234" / "1,234,567" / "257,565"
            #   2. AAA(,BB)?     → angka biasa atau "12.34" / "12,34" (decimal 1-2 digit)
            money_pattern = re.compile(r'\d{1,3}(?:[,.]\d{3})+|\d+(?:[.,]\d{1,2})?')
            numbers = money_pattern.findall(text_clean)
            numbers = [n.replace(' ', '') for n in numbers]
            # Special case: "25723,415565" → setelah money_pattern menangkap "25723" dan ",415565"
            # adalah artifact OCR concat 2 angka. Cek: jika 2 angka berurutan dan
            # gabungan-nya membentuk pola "AAA,BBB" + "CCC,DDD" yang masuk akal, split ulang.
            if len(numbers) == 2:
                a, b = numbers[0].replace(' ', ''), numbers[1].replace(' ', '')
                # Heuristik: gabungan tanpa koma adalah 2 angka serupa panjangnya
                joined = (a + b).replace(',', '').replace('.', '')
                if len(joined) % 2 == 0 and len(joined) >= 8:
                    half = len(joined) // 2
                    p1, p2 = joined[:half], joined[half:]
                    # Reformat dengan koma ribuan
                    def _fmt(s):
                        return f"{int(s):,}".replace(',', ',')
                    try:
                        v1, v2 = int(p1), int(p2)
                        # Heuristik: keduanya harus dalam rentang masuk akal (>= 1000, <= 100M)
                        if 1000 <= v1 <= 100_000_000 and 1000 <= v2 <= 100_000_000:
                            numbers = [str(v1), str(v2)]
                    except ValueError:
                        pass
            
            # IMPORTANT: Check 'subtotal' BEFORE 'total' (since "subtotal" contains "total")
            is_subtotal = any(k in text_lower for k in ['subtotal', 'sub total', 'sub-total', 'jumlah'])
            is_explicit_grand = any(k in text_lower for k in [
                'grand total', 'total bayar', 'total amount', 'total pembayaran',
                'total akhir', 'total belanja', 'total tagihan', 'nett total', 'net total',
                'total hrg'
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
            is_cash_line = any(k in text_lower for k in ['cash', 'tunai', 'paid', 'jumlah bayar'])
            # "bayar" sendiri bisa berarti "Total Bayar" (grand total) ATAU "Jumlah Bayar" (cash).
            # Jika ada "jumlah" → cash. Jika ada "total" → grand total. Standalone "bayar" → cash.
            if 'bayar' in text_lower and 'total' not in text_lower:
                is_cash_line = True
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
                    # Skip percentage values (e.g. "Tax 10%" / "Tax Resto 10*")
                    # Tax actual value biasanya >= 100; angka kecil di tax line = persentase
                    if val < 100:
                        continue
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
                # Baris angka tanpa keyword yang bukan di kategori manapun → skip
                # (sebelumnya pakai is_total_target dari cascade, sekarang dinonaktifkan)
        
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
        
        # Fallback: jika grand_total masih 0, scan SEMUA baris untuk keyword "total" (bukan subtotal)
        # dan ambil angka dari baris itu sendiri ATAU baris ±2 di sekitarnya yang pure number.
        # EXCLUDE area cash/kembali agar tidak salah ambil jumlah bayar sebagai grand total.
        if totals['grand_total'] == 0.0:
            all_total_numbers = []
            # Identifikasi baris cash/change area
            cash_change_kw = ['cash', 'tunai', 'jumlah bayar', 'kembali', 'kembalian', 'change']
            cash_area_indexes = set()
            for i, line in enumerate(sorted_total_lines):
                if any(k in line['text'].lower() for k in cash_change_kw):
                    # Hanya exclude baris ini + 1 baris setelahnya (value-nya)
                    cash_area_indexes.add(i)
                    cash_area_indexes.add(i + 1)

            for i, line in enumerate(sorted_total_lines):
                if i in cash_area_indexes:
                    continue
                t = line['text'].lower()
                if 'tota' in t and 'subtota' not in t and 'ite' not in t and 'oty' not in t and 'qty' not in t:
                    text_clean = line['text'].replace(' ', '')
                    text_clean = re.sub(r'(\d)\s*([,.])', r'\1\2', text_clean)
                    text_clean = re.sub(r'([,.])\s*(\d)', r'\1\2', text_clean)
                    money_re = re.compile(r'\d{1,3}(?:[,.]\d{3})+|\d+')
                    nums_here = [self._parse_number(n) for n in money_re.findall(text_clean)]
                    all_total_numbers.extend([n for n in nums_here if 1000 <= n <= 100_000_000])
                    
                    for offset in range(-2, 3):
                        j = i + offset
                        if j < 0 or j >= len(sorted_total_lines) or j == i:
                            continue
                        if j in cash_area_indexes:
                            continue
                        neighbor = sorted_total_lines[j]['text'].strip()
                        if re.match(r'^[\d\s,.\-]+$', neighbor):
                            neighbor_clean = re.sub(r'\s+', '', neighbor)
                            neighbor_nums = [self._parse_number(n) for n in money_re.findall(neighbor_clean)]
                            all_total_numbers.extend([n for n in neighbor_nums if 1000 <= n <= 100_000_000])
            
            if all_total_numbers:
                totals['grand_total'] = max(all_total_numbers)

        # Fallback: Calculate grand total if still not found
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
