"""
OCR FinSight - Receipt Data Extractor
Regex-based extraction from classified OCR lines.
"""

import re
from typing import Optional
from app.services.text_cleaner import OCRTextCleaner


class ReceiptExtractor:

    def __init__(self):
        self.cleaner = OCRTextCleaner()

    def extract(self, classified_lines: list[dict]) -> dict:
        total_keywords = ['total', 'subtota', 'amount', 'rounding', 'tunai',
                          'kembali', 'change', 'cash', 'bayar']
        for i in range(len(classified_lines)):
            text_lower = classified_lines[i]['text'].lower()
            if any(k in text_lower for k in total_keywords):
                classified_lines[i]['predicted_class'] = 'TOTAL_PAYMENT'

            is_explicit_total = any(k in text_lower for k in ['total', 'subtota', 'amount', 'bayar'])
            if classified_lines[i].get('predicted_class') == 'TOTAL_PAYMENT':
                for offset in range(1, 3):
                    if i + offset < len(classified_lines):
                        target = classified_lines[i + offset]
                        t = target['text'].strip()
                        if any(c.isdigit() for c in t) and len(t) <= 15:
                            if sum(c.isalpha() for c in t) <= 5:
                                target['predicted_class'] = 'TOTAL_PAYMENT'
                                if is_explicit_total:
                                    target['is_total_target'] = True

        grouped = {}
        for line in classified_lines:
            cls = line.get('predicted_class', 'OTHER')
            grouped.setdefault(cls, []).append(line)

        result = {
            'store': self._extract_store(grouped.get('STORE', [])),
            'date': self._extract_date(grouped.get('DATE', [])),
            'items': self._extract_items(
                grouped.get('ITEM_DESC', []) + grouped.get('ITEM_PRICE/QTY', [])),
            'totals': self._extract_total(grouped.get('TOTAL_PAYMENT', [])),
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

    def _extract_store(self, store_lines: list[dict]) -> str:
        if not store_lines:
            return ""
        valid = []
        for line in store_lines:
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

    def _extract_date(self, date_lines: list[dict]) -> str:
        numeric_patterns = [
            r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',
            r'(\d{1,2}[/\-\.]\s*\d{1,2}[/\-\.]\s*\d{2,4})',
        ]
        word_patterns = [
            r'((jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}[\s,]*\d{2,4})',
            r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4})',
            r'(\d{1,2}\s+(jan(uari)?|feb(ruari)?|mar(et)?|apr(il)?|mei|jun(i)?|jul(i)?|agu(stus)?|sep(tember)?|okt(ober)?|nov(ember)?|des(ember)?)\s+\d{2,4})',
        ]
        time_pat = r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)'

        def find_date(text):
            for pat in word_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    date_str = m.group(1).strip()
                    tm = re.search(time_pat, text, re.IGNORECASE)
                    return f"{date_str} {tm.group(1)}".strip() if tm else date_str
            for pat in numeric_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    date_str = re.sub(r'\s+', '', m.group(1).strip())
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
            for line in date_lines:
                if line['text'].strip():
                    return line['text'].strip()
        return ""

    def _extract_items(self, item_lines: list[dict]) -> list[dict]:
        if not item_lines:
            return []

        blacklist = [
            'item:', 'qty:', 'quantity:', 'price:', 'amount:', 'total:', 'subtotal:',
            'cashier', 'kasir', 'waiter', 'server', 'customer', 'pelanggan', 'member',
            'table', 'meja', 'receipt', 'struk', 'invoice', 'transaction', 'transaksi',
            'tender', 'payment', 'pembayaran', 'cash', 'tunai', 'change', 'kembali',
            'discount', 'diskon', 'tax', 'pajak', 'gst', 'vat', 'ppn',
            'total', 'subtotal', 'grand total', 'jumlah',
            'point', 'reward', 'thank', 'terima', 'kasih', 'welcome', 'selamat',
            'jl.', 'jalan', 'tel:', 'telp', 'phone', 'www.', '.com',
            'date', 'tanggal', 'time', 'jam', 'sdn bhd', 'pt.', 'cv.',
        ]

        filtered = []
        for line in sorted(item_lines, key=lambda l: l.get('y_min', 0)):
            y = line.get('y_min', 0)
            text = line['text'].strip()
            text_lower = text.lower()
            if y < 0.10 or y > 0.92:
                continue
            if len(text) < 3:
                continue
            if any(kw in text_lower for kw in blacklist):
                continue
            if re.match(r'^[\d\s\.,\*\-\/x@xX]+$', text):
                continue
            if text.upper() in ['RM', 'RP', 'IDR', 'SR', '$', 'USD']:
                continue
            if sum(c.isalpha() for c in text) < 3:
                continue
            filtered.append(line)

        if not filtered:
            return []

        merged_groups, current, last_y = [], [], None
        for line in filtered:
            y = line.get('y_min', 0)
            if last_y is not None and abs(y - last_y) < 0.025:
                current.append(line)
            else:
                if current:
                    merged_groups.append(current)
                current = [line]
            last_y = y
        if current:
            merged_groups.append(current)

        items = []
        for group in merged_groups:
            group_sorted = sorted(group, key=lambda l: l.get('x_min', 0))
            combined = " ".join(l['text'].strip() for l in group_sorted)
            item = self._parse_item_line(combined)
            if item and item['name'] and sum(c.isalpha() for c in item['name']) >= 3:
                items.append(item)
        return items

    def _parse_item_line(self, text: str) -> Optional[dict]:
        if not text or len(text) < 3:
            return None
        text = self.cleaner.clean_item(text)
        item = {'name': '', 'qty': 1, 'price': 0.0, 'raw': text}

        m = re.search(r'(\d+)\s*[xX\*]\s*([\d.,]+)', text)
        if m:
            item['qty'] = int(m.group(1))
            item['price'] = self._parse_number(m.group(2))
            item['name'] = text[:m.start()].strip()
            return item if item['name'] else None

        for pat, has_qty in [
            (r'([\d.,]+)\s*\*\s*(\d+)', True),
            (r'([\d.,]+)\s+(?:SR|RM|IDR|Rp|rp)', False),
            (r'@\s*([\d.,]+)', False),
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                price = self._parse_number(m.group(1))
                if price > 0:
                    item['price'] = price
                    if has_qty and len(m.groups()) > 1:
                        try:
                            item['qty'] = int(m.group(2))
                        except (ValueError, IndexError):
                            pass
                    item['name'] = (text[:m.start()] or text[m.end():]).strip()
                    return item if item['name'] else None

        m = re.search(r'\s+([\d.,]{3,})\s*$', text)
        if m:
            price = self._parse_number(m.group(1))
            if 50 <= price < 10_000_000:
                item['price'] = price
                item['name'] = text[:m.start()].strip()
                return item if item['name'] else None

        m = re.search(r'(?:Rp|RM|IDR|rp)\s*([\d.,]+)', text, re.IGNORECASE)
        if m:
            price = self._parse_number(m.group(1))
            if price >= 50:
                item['price'] = price
                item['name'] = (text[:m.start()] + " " + text[m.end():]).strip()
                return item if item['name'] else None

        alpha = sum(c.isalpha() for c in text)
        digit = sum(c.isdigit() for c in text)
        if alpha > digit and alpha >= 3:
            item['name'] = text
            return item
        return None

    def _extract_total(self, total_lines: list[dict]) -> dict:
        totals = {'grand_total': 0.0, 'subtotal': 0.0, 'discount': 0.0,
                  'tax': 0.0, 'cash': 0.0, 'change': 0.0}
        grand_cands, sub_cands, disc_cands, tax_cands, cash_cands, chg_cands = [], [], [], [], [], []

        for line in sorted(total_lines, key=lambda l: l.get('y_min', 0)):
            text = line['text']
            tl = text.lower()

            if any(k in tl for k in ['npwp', 'tel', 'fax', 'phone', 'call', 'roc', 'gst no', 'trxid', 'member']):
                continue
            if len(re.findall(r'\d', text)) >= 10 and (text.count('-') >= 1 or text.count('.') >= 2):
                continue

            nums = re.findall(r'[\d]+[.,]?[\d]*', text.replace('O', '0').replace('o', '0'))
            is_sub = any(k in tl for k in ['subtotal', 'sub total', 'sub-total', 'jumlah'])
            is_grand = any(k in tl for k in [
                'grand total', 'total bayar', 'total amount', 'total pembayaran',
                'total akhir', 'total belanja', 'nett total', 'net total'])
            is_service = any(k in tl for k in ['service charge', 'service', 'charge', 'biaya'])
            is_tax = any(k in tl for k in ['tax', 'pajak', 'ppn', 'gst', 'vat', 'pb1'])
            is_disc = any(k in tl for k in ['discount', 'diskon', 'potongan', 'disc', 'voucher', 'promo'])
            is_cash = any(k in tl for k in ['cash', 'tunai', 'bayar', 'paid'])
            is_chg = any(k in tl for k in ['change', 'kembali', 'kembalian'])
            is_total = ('total' in tl and not is_sub and not is_service
                        and not is_tax and not is_disc and not is_cash and not is_chg)

            for num_str in nums:
                val = self._parse_number(num_str)
                if val <= 0 or val > 100_000_000:
                    continue
                if is_grand:
                    grand_cands.append((val, 10))
                elif is_sub:
                    sub_cands.append(val)
                elif is_tax:
                    tax_cands.append(val)
                elif is_disc:
                    disc_cands.append(val)
                elif is_cash:
                    cash_cands.append(val)
                elif is_chg:
                    chg_cands.append(val)
                elif is_total:
                    grand_cands.append((val, 5))
                elif line.get('is_total_target'):
                    grand_cands.append((val, 3))

        if sub_cands:   totals['subtotal'] = max(sub_cands)
        if disc_cands:  totals['discount'] = max(disc_cands)
        if tax_cands:   totals['tax'] = max(tax_cands)
        if cash_cands:  totals['cash'] = max(cash_cands)
        if chg_cands:   totals['change'] = max(chg_cands)

        if grand_cands:
            grand_cands.sort(key=lambda x: (x[1], x[0]), reverse=True)
            totals['grand_total'] = grand_cands[0][0]

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
            parts = num_str.split('.')
            if len(parts[-1]) == 3:
                num_str = num_str.replace('.', '')
        try:
            return float(num_str)
        except ValueError:
            return 0.0
