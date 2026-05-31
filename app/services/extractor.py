"""
OCR FinSight - Receipt Data Extractor
Regex-based extraction from classified OCR lines.
"""

import re
import numpy as np
from typing import Optional
from app.services.text_cleaner import OCRTextCleaner


class ReceiptExtractor:

    def __init__(self):
        self.cleaner = OCRTextCleaner()

    def extract(self, classified_lines: list[dict]) -> dict:
        total_substrings = ['total', 'subtota', 'amount', 'rounding', 'tunai', 'kembali', 'bayar']
        total_word_re = re.compile(r'\b(cash|change)\b', re.IGNORECASE)
        total_exclusions_re = re.compile(r'\b(cashier|kasir|server|waiter|operator)\b', re.IGNORECASE)

        for i in range(len(classified_lines)):
            text = classified_lines[i]['text']
            text_lower = text.lower()
            if total_exclusions_re.search(text):
                continue
            if any(k in text_lower for k in total_substrings) or total_word_re.search(text):
                classified_lines[i]['predicted_class'] = 'TOTAL_PAYMENT'

        grouped = {}
        for line in classified_lines:
            cls = line.get('predicted_class', 'OTHER')
            grouped.setdefault(cls, []).append(line)

        result = {
            'store': self._extract_store(grouped.get('STORE', []), classified_lines),
            'date': self._extract_date(grouped.get('DATE', []), classified_lines),
            'items': self._extract_items(classified_lines),
            'totals': self._extract_total(classified_lines),
            'total': 0.0,
            'address': self._extract_address(grouped.get('ADDRESS_CONTACT', [])),
            'raw_lines': [
                {
                    'text': l['text'],
                    'class': l.get('predicted_class', 'OTHER'),
                    'confidence': float(l.get('class_confidence', 0.0)),
                    'bbox': [[float(pt[0]), float(pt[1])] for pt in l['bbox']] if 'bbox' in l else None,
                    'x_min': float(l.get('x_min', 0)), 'y_min': float(l.get('y_min', 0)),
                    'width': float(l.get('width', 0)), 'height': float(l.get('height', 0)),
                }
                for l in classified_lines
            ],
        }
        result['total'] = result['totals']['grand_total']
        return result

    def _extract_store(self, store_lines: list[dict], all_lines: list[dict] | None = None) -> str:
        """Extract store name. Falls back to top ADDRESS_CONTACT/OTHER if STORE label is empty."""
        candidates = list(store_lines) if store_lines else []
        if not candidates and all_lines:
            candidates = [l for l in all_lines if l.get('predicted_class') in ('ADDRESS_CONTACT', 'OTHER')]

        if not candidates:
            return ""

        valid = []
        for line in candidates:
            text = line['text'].strip()
            if not text or len(text) < 3:
                continue
            alpha = sum(c.isalpha() for c in text)
            digit = sum(c.isdigit() for c in text)
            if alpha < 3 or digit > alpha:
                continue
            if re.search(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', text):
                continue
            if re.match(r'^[\d\s\.,\*\-]+$', text):
                continue
            if re.match(r'^(tel|phone|fax|hp)[\s:.]', text, re.IGNORECASE):
                continue
            valid.append(line)

        if not valid:
            return ""

        parts = []
        for line in sorted(valid, key=lambda l: l.get('y_min', 0))[:3]:
            cleaned = self.cleaner.clean_store(line['text'].strip())
            if cleaned and len(cleaned) > 2:
                parts.append(cleaned)
        return ' '.join(parts).title() if parts else ""

    def _extract_date(self, date_lines: list[dict], all_lines: list[dict] | None = None) -> str:
        """Extract date. Supports numeric, word-month (EN/ID), and compact formats.
        Falls back to scanning all lines if classifier missed the date label.
        """
        numeric_patterns = [
            (r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', 'iso'),
            (r'(\d{1,2})[/\-\.]\s*(\d{1,2})[/\-\.]\s*(\d{2,4})', 'dmy'),
        ]
        word_patterns = [
            r'((jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}[\s,]*\d{4})',
            r'((jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*\d{1,2}\s*[,.]?\s*\d{4})',
            r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4})',
            r'(\d{1,2}[\-\.](jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\-\.]\d{4})',
            r'(\d{1,2}\s+(jan(uari)?|feb(ruari)?|mar(et)?|apr(il)?|mei|jun(i)?|jul(i)?|agu(stus)?|sep(tember)?|okt(ober)?|nov(ember)?|des(ember)?)\s+\d{4})',
        ]
        time_pat = r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)'

        def is_valid_dmy(d, m, y):
            return 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= (y if y >= 100 else 2000 + y) <= 2100

        def find_date(text):
            for pat in word_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    date_str = m.group(1).strip()
                    tm = re.search(time_pat, text, re.IGNORECASE)
                    return f"{date_str} {tm.group(1)}".strip() if tm else date_str
            for pat, kind in numeric_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    if kind == 'iso':
                        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    else:
                        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    if not is_valid_dmy(d, mo, y):
                        continue
                    date_str = re.sub(r'\s+', '', m.group(0).strip())
                    tm = re.search(time_pat, text, re.IGNORECASE)
                    return f"{date_str} {tm.group(1)}".strip() if tm else date_str
            return ""

        if date_lines:
            for line in sorted(date_lines, key=lambda l: l.get('y_min', 0)):
                t = line['text'].strip()
                if re.match(r'^\s*\d{1,2}:\d{2}', t) and not re.search(r'\d{4}|\d{1,2}[/\-\.]', t):
                    continue
                if re.match(r'^\s*(printed|cetak|dicetak)', t, re.IGNORECASE):
                    continue
                result = find_date(t)
                if result:
                    return result
            result = find_date(" ".join(l['text'].strip() for l in date_lines))
            if result:
                return result

        # Fallback: scan all lines with a proximity window
        if all_lines:
            sorted_all = sorted(all_lines, key=lambda l: l.get('y_min', 0))
            Y_TOL = 0.03
            for i in range(len(sorted_all)):
                for window in (1, 2, 3):
                    chunk = sorted_all[i:i + window]
                    if len(chunk) < window:
                        continue
                    y_vals = [l.get('y_min', 0) for l in chunk]
                    if max(y_vals) - min(y_vals) > Y_TOL:
                        continue
                    joined = " ".join(l['text'].strip() for l in chunk)
                    if re.match(r'^\s*(printed|cetak|dicetak)', joined, re.IGNORECASE):
                        continue
                    result = find_date(joined)
                    if result:
                        return result

        if date_lines:
            for line in sorted(date_lines, key=lambda l: l.get('y_min', 0)):
                if line['text'].strip():
                    return line['text'].strip()
        return ""

    def _extract_items(self, all_lines: list[dict]) -> list[dict]:
        """Price-driven column-aware item extraction.

        Splits lines into NAME (left column, x_min < 0.55) and PRICE (right column, x_min >= 0.55),
        then pairs each price line with the nearest name line by y-overlap.
        """
        if not all_lines:
            return []

        blacklist = [
            'item:', 'qty:', 'quantity:', 'price:', 'amount:', 'total:', 'subtotal:',
            'cashier', 'kasir', 'waiter', 'waitress', 'server', 'staff', 'operator',
            'customer', 'pelanggan', 'member', 'membership', 'card no',
            'table', 'meja', 'pax', 'guest', 'room',
            'receipt', 'struk', 'bill', 'invoice', 'transaction', 'transaksi',
            'order', 'pesanan', 'ref', 'reference',
            'tender', 'payment', 'pembayaran', 'cash', 'tunai', 'card', 'credit',
            'debit', 'change', 'kembali', 'kembalian',
            'discount', 'diskon', 'potongan', 'promo', 'voucher', 'coupon',
            'service charge', 'tax', 'pajak', 'gst', 'vat', 'ppn',
            'total', 'subtotal', 'grand total', 'amount', 'jumlah',
            'point', 'points', 'reward', 'saving', 'hemat', 'earned',
            'thank', 'terima', 'kasih', 'welcome', 'selamat', 'datang',
            'please', 'silakan', 'come again', 'visit',
            'reg', 'register', 'void', 'cancel', 'refund', 'return',
            'open', 'close', 'shift', 'balance', 'saldo',
            'jl.', 'jalan', 'tel:', 'telp', 'phone', 'fax', 'email', 'www.', '.com',
            'date', 'tanggal', 'tgl', 'time', 'jam',
            'sdn bhd', 'pt.', 'cv.',
            'http', 'instagram', 'facebook', 'whatsapp',
        ]

        def is_blacklisted(text):
            return any(k in text.lower() for k in blacklist)

        zone = [l for l in all_lines if 0.18 <= l.get('y_min', 0) <= 0.75
                and not is_blacklisted(l['text'])]
        if not zone:
            return []

        price_re = re.compile(r'\d')
        money_re = re.compile(r'\d{1,3}(?:[,.]\d{3})+|\d+')
        unit_re = re.compile(r'^\d*\s*(pcs|btl|bks|kg|gr|ml|ltr|dus|box|set|pack|unit)\s*@?\s*$', re.IGNORECASE)

        name_lines, price_lines = [], []
        for l in zone:
            text = l['text'].strip()
            if not text:
                continue
            x_min = l.get('x_min', 0)
            if x_min >= 0.55 and price_re.search(text):
                nums = money_re.findall(text.replace(' ', ''))
                if any(self._parse_number(n) >= 100 for n in nums):
                    price_lines.append(l)
            else:
                if re.match(r'^[\d\s\.,\*\-\/x@xX]+$', text):
                    continue
                if text.upper() in ['RM', 'RP', 'IDR', 'SR', '$', 'USD']:
                    continue
                if sum(c.isalpha() for c in text) < 2:
                    continue
                if text.strip().upper() in ['PCS', 'BTL', 'BKS', 'KG', 'GR', 'ML', 'LTR', 'DUS', 'BOX', 'SET', 'PACK', 'UNIT']:
                    continue
                if unit_re.match(text.strip()):
                    continue
                name_lines.append(l)

        if not price_lines:
            return []

        all_heights = [l.get('height', 0.02) for l in name_lines + price_lines]
        avg_h = sum(all_heights) / len(all_heights) if all_heights else 0.02
        Y_TOL = max(avg_h * 1.0, 0.012)

        items = []
        used = set()
        for pl in sorted(price_lines, key=lambda l: l.get('y_min', 0)):
            p_y = (pl.get('y_min', 0) + pl.get('y_max', 0)) / 2
            nums = money_re.findall(pl['text'].replace(' ', ''))
            price_val = 0.0
            for n in nums:
                v = self._parse_number(n)
                if 100 <= v <= 10_000_000:
                    price_val = v
                    break
            if price_val <= 0:
                continue

            best_name, best_dist, best_idx = None, float('inf'), -1
            for idx, nl in enumerate(name_lines):
                if idx in used:
                    continue
                n_y = (nl.get('y_min', 0) + nl.get('y_max', 0)) / 2
                if n_y > p_y + Y_TOL * 0.3:
                    continue
                dist = abs(n_y - p_y)
                if dist <= Y_TOL and dist < best_dist:
                    best_dist, best_name, best_idx = dist, nl, idx

            if best_name is None:
                continue
            used.add(best_idx)

            cleaned = self.cleaner.clean_item(best_name['text'].strip()).strip()
            if not cleaned or sum(c.isalpha() for c in cleaned) < 3:
                continue
            if any(k in cleaned.lower() for k in ['tota', 'subtota', 'service', 'printed']):
                continue

            items.append({'name': cleaned, 'qty': 1, 'price': float(price_val), 'raw': best_name['text'].strip()})

        return items

    def _extract_total(self, all_lines: list[dict]) -> dict:
        """Extract totals from TOTAL_PAYMENT-labeled lines using keyword priority."""
        totals = {'grand_total': 0.0, 'subtotal': 0.0, 'discount': 0.0,
                  'tax': 0.0, 'cash': 0.0, 'change': 0.0}
        grand_cands, sub_cands, disc_cands, tax_cands, cash_cands, chg_cands = [], [], [], [], [], []

        total_lines = [l for l in all_lines if l.get('predicted_class') == 'TOTAL_PAYMENT']
        sorted_lines = sorted(total_lines, key=lambda l: l.get('y_min', 0))

        money_re = re.compile(r'\d{1,3}(?:[,.]\d{3})+|\d+(?:[.,]\d{1,2})?')

        for line in sorted_lines:
            text = line['text']
            tl = text.lower()

            has_kw = any(k in tl for k in [
                'total', 'subtota', 'amount', 'service', 'charge', 'tax', 'pajak',
                'discount', 'diskon', 'cash', 'tunai', 'bayar', 'change', 'kembali',
                'rounding', 'ppn', 'gst', 'vat',
            ])
            is_pure_num = bool(re.match(r'^[\d\s,.\-]+$', text.strip()))
            if not has_kw and not is_pure_num:
                continue
            if any(k in tl for k in ['npwp', 'tel', 'fax', 'phone', 'roc', 'gst no', 'trxid', 'member']):
                continue
            if len(re.findall(r'\d', text)) >= 10 and (text.count('-') >= 1 or text.count('.') >= 2):
                continue

            tc = re.sub(r'(\d)\s+([,.])', r'\1\2', text.replace('O', '0').replace('o', '0'))
            tc = re.sub(r'([,.])\s+(\d)', r'\1\2', tc)
            numbers = [n.replace(' ', '') for n in money_re.findall(tc)]

            is_sub = any(k in tl for k in ['subtotal', 'sub total', 'sub-total', 'jumlah'])
            is_grand = any(k in tl for k in [
                'grand total', 'total bayar', 'total amount', 'total pembayaran',
                'total akhir', 'total belanja', 'total tagihan', 'nett total', 'net total', 'total hrg'])
            is_service = any(k in tl for k in ['service charge', 'service', 'charge', 'biaya'])
            is_tax = any(k in tl for k in ['tax', 'pajak', 'ppn', 'gst', 'vat', 'pb1'])
            is_disc = any(k in tl for k in ['discount', 'diskon', 'potongan', 'disc', 'voucher', 'promo'])
            is_cash = any(k in tl for k in ['cash', 'tunai', 'paid', 'jumlah bayar'])
            if 'bayar' in tl and 'total' not in tl:
                is_cash = True
            is_chg = any(k in tl for k in ['change', 'kembali', 'kembalian'])
            is_total = ('total' in tl and not is_sub and not is_service
                        and not is_tax and not is_disc and not is_cash and not is_chg)

            for num_str in numbers:
                val = self._parse_number(num_str)
                if val <= 0 or val > 100_000_000:
                    continue
                if is_grand:            grand_cands.append((val, 10))
                elif is_sub:            sub_cands.append(val)
                elif is_service:        pass
                elif is_tax:
                    if val >= 100:      tax_cands.append(val)
                elif is_disc:           disc_cands.append(val)
                elif is_cash:           cash_cands.append(val)
                elif is_chg:            chg_cands.append(val)
                elif is_total:          grand_cands.append((val, 5))

        if sub_cands:   totals['subtotal'] = max(sub_cands)
        if disc_cands:  totals['discount'] = max(disc_cands)
        if tax_cands:   totals['tax'] = max(tax_cands)
        if cash_cands:  totals['cash'] = max(cash_cands)
        if chg_cands:   totals['change'] = max(chg_cands)

        if grand_cands:
            grand_cands.sort(key=lambda x: (x[1], x[0]), reverse=True)
            totals['grand_total'] = grand_cands[0][0]

        # Fallback: keyword scan across all total lines
        if totals['grand_total'] == 0.0:
            cash_kw = ['cash', 'tunai', 'jumlah bayar', 'kembali', 'kembalian', 'change']
            cash_idx = set()
            for i, line in enumerate(sorted_lines):
                if any(k in line['text'].lower() for k in cash_kw):
                    cash_idx.update([i, i + 1])

            candidates = []
            for i, line in enumerate(sorted_lines):
                if i in cash_idx:
                    continue
                t = line['text'].lower()
                if 'tota' in t and 'subtota' not in t and 'qty' not in t:
                    tc2 = re.sub(r'\s+', '', line['text'])
                    tc2 = re.sub(r'(\d)\s*([,.])', r'\1\2', tc2)
                    nums2 = [self._parse_number(n) for n in money_re.findall(tc2)]
                    candidates.extend(n for n in nums2 if 1_000 <= n <= 100_000_000)
            if candidates:
                totals['grand_total'] = max(candidates)

        # Arithmetic fallback
        if totals['grand_total'] == 0.0:
            if totals['subtotal'] > 0:
                totals['grand_total'] = max(0.0, totals['subtotal'] + totals['tax'] - totals['discount'])
            elif totals['cash'] > 0 and totals['change'] > 0:
                totals['grand_total'] = totals['cash'] - totals['change']

        if (totals['grand_total'] > 0 and totals['subtotal'] > 0
                and totals['grand_total'] < totals['subtotal']):
            totals['grand_total'] = totals['subtotal'] + totals['tax'] - totals['discount']

        return totals

    def _extract_address(self, address_lines: list[dict]) -> str:
        if not address_lines:
            return ""
        return ", ".join(self.cleaner.clean_address(l['text'].strip()) for l in address_lines)

    def _parse_number(self, num_str: str) -> float:
        num_str = num_str.strip()
        if not num_str:
            return 0.0
        if '.' in num_str and ',' in num_str:
            num_str = num_str.replace('.', '').replace(',', '.')
        elif ',' in num_str:
            parts = num_str.split(',')
            num_str = num_str.replace(',', '.') if len(parts[-1]) == 2 else num_str.replace(',', '')
        elif '.' in num_str:
            if len(num_str.split('.')[-1]) == 3:
                num_str = num_str.replace('.', '')
        try:
            return float(num_str)
        except ValueError:
            return 0.0
