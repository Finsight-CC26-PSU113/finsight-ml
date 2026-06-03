"""
OCR FinSight - Receipt Data Extractor
Regex-based value extraction dari classified OCR lines.
"""

import re
import numpy as np
from typing import Optional
from src.text_cleaner import OCRTextCleaner
from src.classifier_corrector import ClassifierCorrector


class ReceiptExtractor:
    """Extract structured data dari baris OCR yang sudah di-classify."""
    
    def __init__(self, use_corrector: bool = True):
        self.cleaner = OCRTextCleaner()
        self.corrector = ClassifierCorrector() if use_corrector else None
    
    def extract(self, classified_lines: list[dict]) -> dict:
        """Extract semua field dari classified lines.
        
        Args:
            classified_lines: List of dicts dengan key 'text' dan 'predicted_class'
            
        Returns:
            Dict berisi: store, date, items, total, raw_lines
        """
        # Apply rule-based corrections FIRST to fix classifier errors
        if self.corrector:
            classified_lines = self.corrector.correct(classified_lines)
        
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
            'debit', 'change', 'kembali', 'kembalian', 'bayar', 'dibayar', 'bayar dengan',
            'tagihan', 'tagih', 'charged', 'due',  # Added: tagihan-related
            'discount', 'diskon', 'potongan', 'promo', 'voucher', 'coupon',
            'service charge', 'tax', 'pajak', 'gst', 'vat', 'ppn', 'pb1',
            'total', 'subtotal', 'grand total', 'amount', 'jumlah', 'jml',
            'harga jual', 'harga', 'total item', 'total qty',
            'rounding', 'pembulatan', 'adjust', 'penyesuaian',  # Added: rounding-related
            'point', 'points', 'reward', 'saving', 'hemat', 'earned',
            'thank', 'terima', 'kasih', 'welcome', 'selamat', 'datang',
            'please', 'silakan', 'come again', 'visit',
            'goods', 'barang', 'product', 'produk',
            'reg', 'register', 'void', 'cancel', 'refund', 'return', 'retur',
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
    
    def _extract_total(self, all_lines: list[dict]) -> dict:
        """Extract multiple totals dengan KEYWORD-FIRST strategy untuk robustness.
        
        Strategy: Keywords lebih reliable daripada classifier untuk total detection.
        Classifier hanya digunakan sebagai fallback.
        
        Returns:
            Dict dengan keys: grand_total, subtotal, discount, tax, service_charge, cash, change
        """
        totals = {
            'grand_total': 0.0,
            'subtotal': 0.0,
            'discount': 0.0,
            'tax': 0.0,
            'service_charge': 0.0,
            'cash': 0.0,
            'change': 0.0
        }
        
        # Kategorisasi berdasarkan KEYWORD FIRST, classifier sebagai fallback
        grand_total_candidates = []
        subtotal_candidates = []
        discount_candidates = []
        tax_candidates = []
        service_charge_candidates = []
        cash_candidates = []
        change_candidates = []

        # Sort by y_min so later lines (typically grand total) win on tie
        sorted_lines = sorted(all_lines, key=lambda l: l.get('y_min', 0))
        
        # Calculate item prices for validation (used in multiple places)
        item_prices = []
        for line in sorted_lines:
            if line.get('predicted_class') == 'ITEM_PRICE/QTY':
                text = line['text'].replace(' ', '')
                money_re = re.compile(r'\d{1,3}(?:[,.]\d{3})+|\d+')
                nums = [self._parse_number(n) for n in money_re.findall(text)]
                item_prices.extend([n for n in nums if 100 <= n <= 10_000_000])

        for line in sorted_lines:
            text = line['text']
            text_lower = text.lower()
            predicted_class = line.get('predicted_class', 'OTHER')

            # Skip lines that are clearly not totals
            if predicted_class in ['STORE', 'ADDRESS_CONTACT', 'DATE', 'ITEM_DESC']:
                continue
            
            # Safeguard: Skip NPWP, phone numbers, transaction IDs
            if any(k in text_lower for k in ['npwp', 'tel', 'fax', 'phone', 'call', 'roc', 'gst no', 'trxid', 'member']):
                continue
            
            # Clean OCR typos
            text_clean = text.replace('O', '0').replace('o', '0')
            text_clean = re.sub(r'(\d)\s+([,.])', r'\1\2', text_clean)
            text_clean = re.sub(r'([,.])\s+(\d)', r'\1\2', text_clean)
            
            # Extract numbers
            money_pattern = re.compile(r'\d{1,3}(?:[,.]\d{3})+|\d+(?:[.,]\d{1,2})?')
            numbers = money_pattern.findall(text_clean)
            numbers = [n.replace(' ', '') for n in numbers]
            
            # KEYWORD-FIRST STRATEGY: Keywords are more reliable than classifier for totals
            # Enhanced keyword matching with more variations
            is_subtotal_kw = any(k in text_lower for k in [
                'subtotal', 'sub total', 'sub-total', 'jumlah', 'sub ttl', 'sub.total',
                'amount', 'amt', 'jml'
            ])
            
            # CRITICAL: "Total Tagihan" (GRAND_TOTAL) vs "Total Bayar" (CASH_PAYMENT)
            # "Tagihan" means "bill/charge" → GRAND_TOTAL
            # "Bayar" with "Total" means "payment amount" → CASH_PAYMENT
            # Priority keywords (most specific first):
            is_grand_total_kw = any(k in text_lower for k in [
                'total tagihan', 'tagihan', 'grand total', 'nett total', 'net total',
                'total akhir', 'total belanja', 'total amount', 'total hrg', 'total harga',
                'amount due', 'balance due', 'total due', 
                'ttl tagihan', 'jumlah tagihan'
            ])
            
            # Generic "total" keyword - ONLY if no "bayar"/"tunai" word nearby
            # This prevents "Total Bayar" from being detected as GRAND_TOTAL
            is_generic_total_kw = (
                ('total' in text_lower or 'ttl' in text_lower or 'tota1' in text_lower)
                and not any(pay_word in text_lower for pay_word in ['bayar', 'tunai', 'cash', 'paid', 'payment', 'tender'])
            )
            
            # Combine both
            is_grand_total_kw = is_grand_total_kw or is_generic_total_kw
            
            is_tax_kw = any(k in text_lower for k in [
                'tax', 'pajak', 'ppn', 'gst', 'vat', 'pb1', 'sst',
                'add gst', 'add tax', 'gst/tax', 'tax/gst', 'cukai',
                'sales tax', 'service tax', 'govt tax'
            ])
            is_discount_kw = any(k in text_lower for k in [
                'discount', 'diskon', 'potongan', 'disc', 'voucher', 'promo',
                'member disc', 'item disc', 'cashback', 'rebate', 'kupon',
                'disct', 'discnt', 'dscnt', 'pot.', 'less', 'saving', 'hemat'
            ])
            is_service_kw = any(k in text_lower for k in [
                'service charge', 'service', 'charge', 'biaya layanan', 'biaya',
                'svc charge', 'svc chg', 'srv charge'
            ])
            is_cash_kw = any(k in text_lower for k in [
                'total bayar', 'bayar', 'dibayar', 'dibayarkan',  # Added: total bayar is CASH
                'cash', 'tunai', 'paid', 'jumlah bayar', 'payment',
                'tender', 'received', 'terima', 'uang diterima'
            ])
            is_change_kw = any(k in text_lower for k in [
                'change', 'kembali', 'kembalian', 'balance', 'return', 'uang kembali'
            ])
            
            for num_str in numbers:
                val = self._parse_number(num_str)
                if val <= 0 or val > 100000000:
                    continue

                # KEYWORD-FIRST: If keyword matches, trust it over classifier
                # This is more robust because keywords for totals are very specific
                if is_subtotal_kw and not is_grand_total_kw:
                    subtotal_candidates.append(val)
                elif is_tax_kw and not (is_subtotal_kw or is_grand_total_kw or is_discount_kw):
                    # Skip percentage values
                    is_percentage = val < 50 and ('%' in text or 'persen' in text_lower)
                    if not is_percentage:
                        tax_candidates.append(val)
                elif is_discount_kw and not (is_subtotal_kw or is_grand_total_kw or is_tax_kw):
                    discount_candidates.append(val)
                elif is_service_kw and not (is_subtotal_kw or is_grand_total_kw or is_tax_kw or is_discount_kw):
                    service_charge_candidates.append(val)
                elif is_grand_total_kw:
                    # Priority: explicit grand total keyword > generic total
                    priority = 10 if any(k in text_lower for k in ['grand', 'bayar', 'akhir', 'nett', 'net', 'due']) else 5
                    grand_total_candidates.append((val, priority))
                elif is_change_kw:
                    change_candidates.append(val)
                elif is_cash_kw:
                    cash_candidates.append(val)
                # FALLBACK: Use classifier prediction if no keyword match
                elif predicted_class == 'SUBTOTAL':
                    subtotal_candidates.append(val)
                elif predicted_class == 'TAX':
                    is_percentage = val < 50 and ('%' in text or 'persen' in text_lower)
                    if not is_percentage:
                        tax_candidates.append(val)
                elif predicted_class == 'DISCOUNT':
                    discount_candidates.append(val)
                elif predicted_class == 'SERVICE_CHARGE':
                    service_charge_candidates.append(val)
                elif predicted_class == 'GRAND_TOTAL':
                    priority = 3  # Lower priority than keyword-based
                    grand_total_candidates.append((val, priority))
                elif predicted_class == 'CASH_PAYMENT':
                    cash_candidates.append(val)
        
        # Assign values with validation
        if subtotal_candidates:
            totals['subtotal'] = max(subtotal_candidates)
        if discount_candidates:
            totals['discount'] = max(discount_candidates)
        if tax_candidates:
            # Validate tax: should be reasonable
            # Tax typically 5-20%, max 100% in extreme cases
            # If tax > items_sum, it's likely misclassified (e.g., grand_total misclassified as tax)
            tax_val = max(tax_candidates)
            items_sum_for_validation = sum(item_prices) if item_prices else 0
            
            if items_sum_for_validation > 0:
                # Tax should be <= items_sum (100% tax is already extreme!)
                if tax_val <= items_sum_for_validation:
                    totals['tax'] = tax_val
                else:
                    print(f"[extractor] ⚠️  Ignoring tax={tax_val:.2f} (> items_sum {items_sum_for_validation:.2f}, likely misclassified)")
            elif totals['subtotal'] > 0:
                # Fallback: use subtotal if no items
                if tax_val <= totals['subtotal']:
                    totals['tax'] = tax_val
            else:
                # No validation possible, accept if reasonable
                if tax_val < 100000:
                    totals['tax'] = tax_val
        if service_charge_candidates:
            # Same validation for service charge
            service_val = max(service_charge_candidates)
            items_sum_for_validation = sum(item_prices) if item_prices else 0
            
            if items_sum_for_validation > 0:
                if service_val <= items_sum_for_validation:
                    totals['service_charge'] = service_val
                else:
                    print(f"[extractor] ⚠️  Ignoring service={service_val:.2f} (> items_sum {items_sum_for_validation:.2f}, likely misclassified)")
            else:
                totals['service_charge'] = service_val

        if cash_candidates:
            totals['cash'] = max(cash_candidates)
        if change_candidates:
            totals['change'] = max(change_candidates)
        
        # Calculate item prices for validation (used in multiple places)
        item_prices = []
        for line in sorted_lines:
            if line.get('predicted_class') == 'ITEM_PRICE/QTY':
                text = line['text'].replace(' ', '')
                money_re = re.compile(r'\d{1,3}(?:[,.]\d{3})+|\d+')
                nums = [self._parse_number(n) for n in money_re.findall(text)]
                item_prices.extend([n for n in nums if 100 <= n <= 10_000_000])
        
        # Grand Total Logic with validation
        if grand_total_candidates:
            grand_total_candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
            candidate_total = grand_total_candidates[0][0]
            
            # Validation rules
            is_valid = True
            
            # Rule 1: Must be >= subtotal (if exists)
            if totals['subtotal'] > 0:
                if candidate_total < totals['subtotal'] * 0.5:
                    is_valid = False
                elif candidate_total > totals['subtotal'] * 2.5:
                    is_valid = False
            
            # Rule 2: Check against items total
            if item_prices:
                estimated_items_total = sum(item_prices)
                if candidate_total < estimated_items_total * 0.3:
                    is_valid = False
                elif candidate_total > estimated_items_total * 3:
                    is_valid = False
            
            # Rule 3: Reject suspicious format (7+ digits without separator)
            candidate_str = str(int(candidate_total))
            if len(candidate_str) >= 7:
                for val, priority in grand_total_candidates[1:]:
                    val_str = str(int(val))
                    if len(val_str) < len(candidate_str):
                        is_valid = False
                        break
            
            if is_valid:
                totals['grand_total'] = candidate_total
            else:
                # Try next candidate
                for val, priority in grand_total_candidates[1:]:
                    valid = True
                    if totals['subtotal'] > 0:
                        if val < totals['subtotal'] * 0.5 or val > totals['subtotal'] * 2.5:
                            valid = False
                    if valid:
                        totals['grand_total'] = val
                        break
        
        # Fallback: calculate from subtotal + tax - discount
        if totals['grand_total'] == 0.0 and totals['subtotal'] > 0:
            calculated = totals['subtotal'] + totals['tax'] + totals['service_charge'] - totals['discount']
            if calculated > 0:
                totals['grand_total'] = calculated
        
        # Fallback 2: use items_total as base if subtotal not found
        if totals['grand_total'] == 0.0 and totals['subtotal'] == 0.0:
            # Calculate from items
            if item_prices:
                estimated_items_total = sum(item_prices)
                # Use items_total as subtotal
                totals['subtotal'] = estimated_items_total
                # Calculate grand_total
                calculated = estimated_items_total + totals['tax'] + totals['service_charge'] - totals['discount']
                if calculated > 0:
                    totals['grand_total'] = calculated
        
        # Last resort: use cash - change
        if totals['grand_total'] == 0.0 and totals['cash'] > 0:
            totals['grand_total'] = totals['cash'] - totals['change']
        
        # ULTIMATE FALLBACK: Scan bottom 30% of receipt for largest number
        # This catches cases where classifier completely fails
        if totals['grand_total'] == 0.0:
            bottom_lines = [l for l in sorted_lines if l.get('y_min', 0) >= 0.7]
            all_numbers = []
            
            for line in bottom_lines:
                # Skip lines with blacklist keywords
                text_lower = line['text'].lower()
                if any(k in text_lower for k in ['npwp', 'tel', 'phone', 'member', 'card', 'cashier', 'kasir']):
                    continue
                
                text = line['text'].replace(' ', '').replace('O', '0').replace('o', '0')
                money_re = re.compile(r'\d{1,3}(?:[,.]\d{3})+|\d+')
                nums = [self._parse_number(n) for n in money_re.findall(text)]
                all_numbers.extend([n for n in nums if 1000 <= n <= 10_000_000])
            
            if all_numbers:
                # Use largest number as grand_total
                largest = max(all_numbers)
                # Validate: should be reasonable compared to items
                if item_prices:
                    items_sum = sum(item_prices)
                    if 0.5 <= largest / max(items_sum, 1) <= 3.0:
                        totals['grand_total'] = largest
                else:
                    totals['grand_total'] = largest
        
        # NO VALIDATION - FULL TRUST OCR + Classifier V4
        # Return whatever was detected, no calculation fallback
        
        if item_prices:
            items_sum = sum(item_prices)
            print(f"[extractor] Items: {len(item_prices)} items, sum={items_sum:.2f}")
            print(f"[extractor] OCR Result: grand_total={totals['grand_total']:.2f}, subtotal={totals['subtotal']:.2f}")
            print(f"[extractor] OCR Result: tax={totals['tax']:.2f}, service={totals['service_charge']:.2f}, discount={totals['discount']:.2f}")
        
        # ONLY calculate if grand_total is completely missing (0.0)
        if totals['grand_total'] == 0.0:
            print(f"[extractor] ⚠️  No grand_total detected, calculating from components")
            if totals['subtotal'] > 0:
                totals['grand_total'] = totals['subtotal'] + totals['tax'] + totals['service_charge'] - totals['discount']
            elif item_prices:
                items_sum = sum(item_prices)
                totals['grand_total'] = items_sum + totals['tax'] + totals['service_charge'] - totals['discount']
                if totals['subtotal'] == 0:
                    totals['subtotal'] = items_sum
            print(f"[extractor] ✅ Calculated grand_total: {totals['grand_total']:.2f}")
        else:
            print(f"[extractor] ✅ Using OCR grand_total: {totals['grand_total']:.2f}")
        
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
        original = num_str
        num_str = num_str.strip()
        if not num_str:
            return 0.0
        
        # Remove currency symbols and spaces
        num_str = re.sub(r'[RMrpRPIDRidrSRsr$€£¥₹]\s*', '', num_str)
        num_str = num_str.strip()
        
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
            # Heuristic: jika semua part (kecuali yang pertama) punya 3 digit → thousands
            # Contoh: "1.699.400" → ['1', '699', '400'] → semua part setelah pertama = 3 digit
            if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
                num_str = num_str.replace('.', '')
            # Jika hanya 1 titik dan part terakhir = 2 digit → decimal
            elif len(parts) == 2 and len(parts[-1]) == 2:
                pass  # Keep as is (decimal)
            # Jika part terakhir = 3 digit → thousands
            elif len(parts[-1]) == 3:
                num_str = num_str.replace('.', '')
        
        try:
            result = float(num_str)
            # Debug log untuk nilai besar yang mencurigakan
            if result > 1000000:
                print(f"⚠️  Large number parsed: '{original}' → {result:,.0f}")
            return result
        except ValueError:
            return 0.0
