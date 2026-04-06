"""Generate synthetic forensic test data simulating a seized hard drive.

Creates ~170 multilingual investigation-relevant files plus ~500 noise files
across categories typical of an international money laundering investigation:
financial documents, communications, media, business records, digital
artifacts, and everyday personal/work clutter.

Content is generated in 8 languages (English, Spanish, Chinese, German, Arabic,
Russian, Japanese, Portuguese) to test the 108-language support of
jina-embeddings-v5-text-nano.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sqlite3
from datetime import datetime, timedelta
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path

import httpx
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
from rich.progress import Progress

from .config import SAMPLE_DRIVE_DIR

log = logging.getLogger(__name__)

LOCALES = ["en_US", "es_MX", "zh_CN", "de_DE", "ar_SA", "ru_RU", "ja_JP", "pt_BR"]
fake = Faker(LOCALES)
Faker.seed(42)
random.seed(42)

# Locale-specific Faker instances for targeted generation
fake_en = Faker("en_US")
fake_es = Faker("es_MX")
fake_zh = Faker("zh_CN")
fake_de = Faker("de_DE")
fake_ar = Faker("ar_SA")
fake_ru = Faker("ru_RU")
fake_ja = Faker("ja_JP")
fake_pt = Faker("pt_BR")
for f in [fake_en, fake_es, fake_zh, fake_de, fake_ar, fake_ru, fake_ja, fake_pt]:
    Faker.seed(42)

LOCALE_FAKERS = {
    "en": fake_en, "es": fake_es, "zh": fake_zh, "de": fake_de,
    "ar": fake_ar, "ru": fake_ru, "ja": fake_ja, "pt": fake_pt,
}

SHELL_COMPANIES = [
    "Pacific Rim Trading Co.", "Meridian Holdings LLC", "Sunrise Capital Partners",
    "Atlas Import Export Ltd.", "Crescent Bay Enterprises", "Golden Gate Ventures Inc.",
    "Pinnacle Asset Management", "Oceanic Resources Group", "Nova Financial Services",
    "Sterling Bridge Consulting", "Harbor Light Industries", "Summit Peak Holdings",
]

SHELL_COMPANIES_INTL = {
    "es": ["Inversiones Sol del Caribe S.A.", "Comercio Marítimo del Pacífico S.L.", "Grupo Empresarial Horizonte"],
    "zh": ["太平洋贸易有限公司", "金桥投资管理集团", "远东资产控股有限公司"],
    "de": ["Alpenland Vermögensverwaltung GmbH", "Rheinische Handelsgesellschaft mbH", "Nordkap Finanz AG"],
    "ar": ["شركة الخليج للاستثمار المحدودة", "مجموعة الشرق للتجارة", "مؤسسة النور المالية"],
    "ru": ["ООО «Балтийская Торговая Компания»", "ЗАО «Восточный Капитал»", "ООО «Сибирские Инвестиции»"],
    "ja": ["太陽商事株式会社", "東洋資産管理合同会社", "桜花キャピタル株式会社"],
    "pt": ["Investimentos Atlântico Ltda.", "Comércio Fluvial do Amazonas S.A.", "Grupo Financeiro do Sul"],
}

ALL_COMPANIES = SHELL_COMPANIES.copy()
for companies in SHELL_COMPANIES_INTL.values():
    ALL_COMPANIES.extend(companies)

OFFSHORE_BANKS = [
    "Cayman National Bank", "Swiss Private Trust AG", "Liechtenstein Landesbank",
    "Singapore Mercantile Bank", "Panama Overseas Banking Corp", "Cyprus Intl Trust",
]

OFFSHORE_BANKS_INTL = {
    "de": ["Zürcher Privatbank AG", "Frankfurter Treuhandbank GmbH"],
    "es": ["Banco Internacional de Panamá", "Banco Privado de las Islas Caimán"],
    "zh": ["新加坡华侨商业银行", "香港远东信托银行"],
    "pt": ["Banco Ultramarino de Macau", "Banco Privado do Atlântico"],
    "ru": ["Балтийский Международный Банк", "Приват-Банк Кипр"],
    "ar": ["بنك الخليج الدولي", "المصرف العربي للتجارة الخارجية"],
}

ALL_BANKS = OFFSHORE_BANKS.copy()
for banks in OFFSHORE_BANKS_INTL.values():
    ALL_BANKS.extend(banks)

CRYPTO_EXCHANGES = [
    "binance.com", "kraken.com", "coinbase.com", "kucoin.com", "bybit.com",
]

PROPERTY_TYPES = [
    "Commercial Office Building", "Luxury Condominium", "Retail Strip Mall",
    "Warehouse Facility", "Residential Duplex", "Beachfront Villa",
]

PROPERTY_TYPES_INTL = {
    "es": ["Condominio de Lujo", "Centro Comercial", "Villa Frente al Mar"],
    "de": ["Luxuswohnung", "Geschäftsgebäude", "Strandvilla"],
    "pt": ["Apartamento de Luxo", "Edifício Comercial", "Casa de Praia"],
    "ja": ["高級マンション", "商業ビル", "海辺のヴィラ"],
    "zh": ["豪华公寓", "商业写字楼", "海滨别墅"],
}


def _random_date(start_year: int = 2021, end_year: int = 2025) -> datetime:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def _pick_locale() -> str:
    return random.choice(list(LOCALE_FAKERS.keys()))


def _locale_company(locale: str) -> str:
    if locale in SHELL_COMPANIES_INTL:
        return random.choice(SHELL_COMPANIES_INTL[locale] + SHELL_COMPANIES[:4])
    return random.choice(SHELL_COMPANIES)


def _locale_bank(locale: str) -> str:
    if locale in OFFSHORE_BANKS_INTL:
        return random.choice(OFFSHORE_BANKS_INTL[locale] + OFFSHORE_BANKS[:2])
    return random.choice(OFFSHORE_BANKS)


def _make_image(text: str, width: int = 640, height: int = 480, bg: str = "white",
                fill: str = "black") -> bytes:
    """Create a simple image with text for testing purposes."""
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except OSError:
            font = ImageFont.load_default()
    lines = text.split("\n")
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=fill, font=font)
        y += 24
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def generate_financial(base: Path, progress, task) -> int:
    """Generate financial documents: spreadsheets, CSVs, invoices in multiple languages."""
    fin_dir = base / "Documents" / "Financial"
    fin_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    currencies_by_locale = {
        "en": ["USD", "GBP"], "es": ["USD", "MXN", "EUR"], "de": ["EUR", "CHF"],
        "zh": ["CNY", "HKD", "USD"], "pt": ["BRL", "USD", "EUR"],
        "ru": ["RUB", "USD", "EUR"], "ar": ["SAR", "AED", "USD"], "ja": ["JPY", "USD"],
    }

    # Transaction ledgers (CSV) -- mixed currencies
    for i in range(8):
        locale = _pick_locale()
        lf = LOCALE_FAKERS[locale]
        path = fin_dir / f"transactions_{2021 + i % 5}_{lf.lexify('???').upper()}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "From", "To", "Amount", "Currency", "Reference", "Type", "Locale"])
            for _ in range(random.randint(50, 200)):
                amt = random.choice([
                    round(random.uniform(100, 9999), 2),
                    round(random.randint(1, 99) * 100, 2),
                ])
                writer.writerow([
                    _random_date().strftime("%Y-%m-%d"),
                    _locale_company(locale),
                    random.choice(ALL_COMPANIES + [lf.company()]),
                    amt,
                    random.choice(currencies_by_locale.get(locale, ["USD", "EUR"])),
                    lf.bothify("TXN-####-????").upper(),
                    random.choice(["Wire Transfer", "ACH", "Check", "Cash Deposit", "SWIFT"]),
                    locale,
                ])
        count += 1
        progress.advance(task)

    # Bank statements (CSV) -- international banks
    for bank in random.sample(ALL_BANKS, min(5, len(ALL_BANKS))):
        path = fin_dir / f"statement_{bank.replace(' ', '_').replace('.', '').replace('/', '_')[:40]}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Description", "Debit", "Credit", "Balance"])
            balance = round(random.uniform(50000, 500000), 2)
            for _ in range(random.randint(30, 80)):
                is_credit = random.random() > 0.4
                amt = round(random.uniform(500, 50000), 2)
                lf = LOCALE_FAKERS[_pick_locale()]
                if is_credit:
                    balance += amt
                    writer.writerow([_random_date().strftime("%Y-%m-%d"), lf.sentence(nb_words=4), "", f"{amt:.2f}", f"{balance:.2f}"])
                else:
                    balance -= amt
                    writer.writerow([_random_date().strftime("%Y-%m-%d"), lf.sentence(nb_words=4), f"{amt:.2f}", "", f"{balance:.2f}"])
        count += 1
        progress.advance(task)

    # Invoices -- multilingual
    inv_dir = fin_dir / "Invoices"
    inv_dir.mkdir(exist_ok=True)

    invoice_templates = {
        "en": ("INVOICE", "From", "To", "Item", "Qty", "Unit Price", "Total", "Payment Terms", "Bank", "Account"),
        "es": ("FACTURA", "De", "Para", "Artículo", "Cant", "Precio Unit", "Total", "Condiciones de Pago", "Banco", "Cuenta"),
        "de": ("RECHNUNG", "Von", "An", "Artikel", "Menge", "Stückpreis", "Gesamt", "Zahlungsbedingungen", "Bank", "Konto"),
        "pt": ("FATURA", "De", "Para", "Item", "Qtd", "Preço Unit", "Total", "Condições de Pagamento", "Banco", "Conta"),
        "ja": ("請求書", "差出人", "宛先", "品目", "数量", "単価", "合計", "支払条件", "銀行", "口座"),
        "zh": ("发票", "寄件人", "收件人", "项目", "数量", "单价", "合计", "付款条款", "银行", "账户"),
    }

    for i in range(15):
        locale = random.choice(list(invoice_templates.keys()))
        lf = LOCALE_FAKERS[locale]
        tmpl = invoice_templates[locale]
        seller = _locale_company(locale)
        buyer = random.choice(ALL_COMPANIES + [lf.company()])
        inv_num = lf.bothify("INV-####-????").upper()
        items = []
        for _ in range(random.randint(1, 5)):
            qty = random.randint(1, 100)
            price = round(random.uniform(50, 5000), 2)
            items.append((lf.bs().title() if hasattr(lf, "bs") else lf.sentence(nb_words=3), qty, price))
        total = sum(q * p for _, q, p in items)
        content = f"""{tmpl[0]}
{tmpl[0]} #: {inv_num}
Date: {_random_date().strftime('%B %d, %Y')}

{tmpl[1]}: {seller}
      {lf.address().replace(chr(10), ', ')}

{tmpl[2]}:   {buyer}
      {lf.address().replace(chr(10), ', ')}

{tmpl[3]:<40} {tmpl[4]:>5} {tmpl[5]:>12} {tmpl[6]:>12}
{'-' * 75}
"""
        for desc, qty, price in items:
            content += f"{desc:<40} {qty:>5} {price:>12.2f} {qty * price:>12.2f}\n"
        content += f"""{'-' * 75}
{tmpl[6]:>57} {total:>12.2f} USD

{tmpl[7]}: Net 30
{tmpl[8]}: {_locale_bank(locale)}
{tmpl[9]}: {lf.iban()}
"""
        (inv_dir / f"{inv_num}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # Tax documents
    for year in range(2021, 2025):
        content = f"""TAX RETURN - YEAR {year}
Filed: April 15, {year + 1}
Taxpayer: {fake_en.name()}
SSN: XXX-XX-{fake_en.numerify('####')}

Gross Income: ${random.randint(80000, 250000):,}
Business Income ({random.choice(SHELL_COMPANIES)}): ${random.randint(50000, 500000):,}
Investment Income: ${random.randint(10000, 100000):,}
Foreign Account Disclosure: {'Yes' if random.random() > 0.3 else 'No'}
FBAR Filed: {'Yes' if random.random() > 0.5 else 'No'}

Total Tax Liability: ${random.randint(20000, 150000):,}
"""
        (fin_dir / f"tax_return_{year}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    return count


def generate_communications(base: Path, progress, task) -> int:
    """Generate email exports, chat logs, and contact lists in multiple languages."""
    comm_dir = base / "Communications"
    comm_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    email_dir = comm_dir / "Email"
    email_dir.mkdir(exist_ok=True)

    subjects_by_locale = {
        "en": [
            "Re: Wire transfer confirmation - {ref}",
            "Meeting tomorrow at {place}",
            "Updated invoice from {company}",
            "Fwd: Account statement {month}",
            "RE: Property closing documents",
            "Need to discuss the {company} arrangement",
            "Urgent: compliance review deadline",
        ],
        "es": [
            "Re: Confirmación de transferencia - {ref}",
            "Reunión mañana en {place}",
            "Factura actualizada de {company}",
            "Re: Estado de cuenta {month}",
            "Urgente: documentos de cierre de propiedad",
        ],
        "de": [
            "AW: Überweisungsbestätigung - {ref}",
            "Treffen morgen in {place}",
            "Aktualisierte Rechnung von {company}",
            "WG: Kontoauszug {month}",
            "Dringend: Compliance-Prüfung Frist",
        ],
        "ja": [
            "Re: 送金確認 - {ref}",
            "明日の会議 - {place}",
            "{company}からの請求書更新",
            "転送: 口座明細 {month}",
        ],
        "zh": [
            "回复：电汇确认 - {ref}",
            "明天在{place}开会",
            "{company}更新发票",
            "转发：账户对账单 {month}",
        ],
        "pt": [
            "Re: Confirmação de transferência - {ref}",
            "Reunião amanhã em {place}",
            "Fatura atualizada de {company}",
            "Enc: Extrato da conta {month}",
        ],
    }

    body_templates_by_locale = {
        "en": [
            "Hi,\n\nPlease find the updated figures for {company}. "
            "The transfer of ${amount:,} has been initiated via {method}.\n\n"
            "Reference: {ref}\n\nRegards,\n{name}",
            "Confirming receipt of payment from {company}. "
            "Amount: ${amount:,}. Please proceed with the next phase.\n\n{name}",
        ],
        "es": [
            "Hola,\n\nAdjunto las cifras actualizadas de {company}. "
            "La transferencia de ${amount:,} ha sido iniciada por {method}.\n\n"
            "Referencia: {ref}\n\nSaludos,\n{name}",
            "Confirmando recepción del pago de {company}. "
            "Monto: ${amount:,}. Proceda con la siguiente fase.\n\n{name}",
        ],
        "de": [
            "Hallo,\n\nAnbei die aktualisierten Zahlen für {company}. "
            "Die Überweisung von ${amount:,} wurde per {method} veranlasst.\n\n"
            "Referenz: {ref}\n\nMit freundlichen Grüßen,\n{name}",
            "Bestätigung des Zahlungseingangs von {company}. "
            "Betrag: ${amount:,}. Bitte fahren Sie mit der nächsten Phase fort.\n\n{name}",
        ],
        "ja": [
            "お疲れ様です。\n\n{company}の最新の数字をお送りします。"
            "${amount:,}の送金は{method}で開始されました。\n\n"
            "参照番号: {ref}\n\nよろしくお願いします。\n{name}",
        ],
        "zh": [
            "您好，\n\n请查收{company}的更新数据。"
            "${amount:,}的转账已通过{method}发起。\n\n"
            "参考编号: {ref}\n\n此致敬礼，\n{name}",
        ],
        "pt": [
            "Olá,\n\nSegue os números atualizados para {company}. "
            "A transferência de ${amount:,} foi iniciada via {method}.\n\n"
            "Referência: {ref}\n\nAtenciosamente,\n{name}",
        ],
    }

    for i in range(30):
        locale = _pick_locale()
        lf = LOCALE_FAKERS[locale]
        locale_key = locale if locale in subjects_by_locale else "en"

        msg = EmailMessage()
        subject_tmpl = random.choice(subjects_by_locale.get(locale_key, subjects_by_locale["en"]))
        subject = subject_tmpl.format(
            ref=lf.bothify("####-????").upper(),
            place=lf.city(),
            company=_locale_company(locale),
            month=fake_en.month_name(),
            q=random.randint(1, 4),
        )
        msg["Subject"] = subject
        msg["From"] = lf.email()
        msg["To"] = lf.email()
        msg["Date"] = _random_date().strftime("%a, %d %b %Y %H:%M:%S +0000")
        msg["X-Language"] = locale
        if random.random() > 0.7:
            msg["Cc"] = lf.email()

        body_tmpl = random.choice(body_templates_by_locale.get(locale_key, body_templates_by_locale["en"]))
        body = body_tmpl.format(
            company=_locale_company(locale),
            amount=random.randint(10000, 500000),
            method=random.choice(["wire transfer", "ACH", "SWIFT"]),
            ref=lf.bothify("TXN-####-????").upper(),
            name=lf.name(),
        )
        msg.set_content(body)
        (email_dir / f"email_{i:03d}.eml").write_text(msg.as_string(), encoding="utf-8")
        count += 1
        progress.advance(task)

    # Chat logs -- multilingual with coded language
    chat_dir = comm_dir / "ChatLogs"
    chat_dir.mkdir(exist_ok=True)

    coded_messages_by_locale = {
        "en": [
            "The package is ready. {company} account.",
            "Move {amount}k to the usual place.",
            "Don't use email for this. Call me.",
            "The lawyer says everything is clean on the {company} side.",
        ],
        "es": [
            "El paquete está listo. Cuenta de {company}.",
            "Mueve {amount}k al lugar de siempre.",
            "No uses correo para esto. Llámame.",
            "El abogado dice que todo está limpio con {company}.",
        ],
        "zh": [
            "包裹准备好了。{company}的账户。",
            "转{amount}万到老地方。",
            "这事别发邮件。给我打电话。",
            "律师说{company}那边没问题。",
        ],
        "ar": [
            "الطرد جاهز. حساب {company}.",
            "حوّل {amount} ألف إلى المكان المعتاد.",
            "لا تستخدم البريد الإلكتروني. اتصل بي.",
            "المحامي يقول كل شيء نظيف مع {company}.",
        ],
        "ru": [
            "Посылка готова. Счёт {company}.",
            "Переведи {amount}k на обычное место.",
            "Не пиши на почту. Позвони мне.",
            "Юрист говорит, что всё чисто по {company}.",
        ],
        "ja": [
            "荷物は準備完了。{company}の口座で。",
            "{amount}万を例の場所に移動して。",
            "メールではやめて。電話して。",
            "弁護士が{company}側は問題ないと言っている。",
        ],
    }

    for i in range(8):
        locale = _pick_locale()
        locale_key = locale if locale in coded_messages_by_locale else "en"
        lf = LOCALE_FAKERS[locale]
        contacts = [(lf.first_name(), lf.phone_number()) for _ in range(3)]
        messages = []
        participants = random.sample(contacts, 2)
        for _ in range(random.randint(15, 50)):
            sender = random.choice(participants)
            msg_tmpl = random.choice(coded_messages_by_locale.get(locale_key, coded_messages_by_locale["en"]))
            messages.append({
                "timestamp": _random_date().isoformat(),
                "sender": sender[0],
                "phone": sender[1],
                "message": msg_tmpl.format(
                    company=_locale_company(locale),
                    amount=random.randint(10, 500),
                ),
                "language": locale,
            })
        messages.sort(key=lambda m: m["timestamp"])
        (chat_dir / f"chat_export_{i:03d}.json").write_text(
            json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        count += 1
        progress.advance(task)

    # Contact lists (.csv) -- international
    path = comm_dir / "contacts.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Phone", "Email", "Organization", "Notes", "Country"])
        for _ in range(40):
            locale = _pick_locale()
            lf = LOCALE_FAKERS[locale]
            writer.writerow([
                lf.name(),
                lf.phone_number(),
                lf.email(),
                random.choice(ALL_COMPANIES + [lf.company(), "", ""]),
                random.choice(["", "", "", "Lawyer", "Accountant", "Banker", "Business partner", "Real estate agent"]),
                lf.country(),
            ])
    count += 1
    progress.advance(task)

    return count


def generate_media(base: Path, progress, task) -> int:
    """Generate images: property photos, receipt scans, screenshots in multiple languages."""
    media_dir = base / "Media" / "Photos"
    media_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = base / "Media" / "Screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # Property photos -- multilingual
    for i in range(12):
        locale = _pick_locale()
        lf = LOCALE_FAKERS[locale]
        prop_list = PROPERTY_TYPES_INTL.get(locale, PROPERTY_TYPES)
        prop = random.choice(prop_list)
        addr = lf.address().replace("\n", ", ")
        text = f"{prop}\n{addr}\nValue: ${random.randint(200000, 5000000):,}\nDate: {_random_date().strftime('%Y-%m-%d')}"
        img_bytes = _make_image(text, bg="#f0f0e8")
        (media_dir / f"property_{i:03d}.jpg").write_bytes(img_bytes)
        count += 1
        progress.advance(task)

    # Receipt scans -- multilingual
    receipt_headers = {
        "en": "RECEIPT", "es": "RECIBO", "de": "QUITTUNG", "pt": "RECIBO",
        "zh": "收据", "ja": "領収書", "ar": "إيصال", "ru": "КВИТАНЦИЯ",
    }
    for i in range(12):
        locale = _pick_locale()
        lf = LOCALE_FAKERS[locale]
        vendor = random.choice([lf.company(), "Cash", _locale_company(locale)])
        amt = round(random.uniform(50, 15000), 2)
        header = receipt_headers.get(locale, "RECEIPT")
        text = f"{header}\n\nVendor: {vendor}\nDate: {_random_date().strftime('%Y-%m-%d')}\nAmount: ${amt:,.2f}\nPayment: {random.choice(['Cash', 'Credit Card', 'Wire'])}\nRef: {lf.bothify('RCP-####')}"
        img_bytes = _make_image(text, 500, 700, bg="#fffef0")
        (media_dir / f"receipt_{i:03d}.jpg").write_bytes(img_bytes)
        count += 1
        progress.advance(task)

    # Banking screenshots
    for i in range(8):
        bank = random.choice(ALL_BANKS)
        balance = random.randint(10000, 2000000)
        text = f"{bank}\nOnline Banking\n\nAccount: ****{fake_en.numerify('####')}\nBalance: ${balance:,}.00\nPending: ${random.randint(1000, 50000):,}.00\n\nRecent Transfer:\n  To: {random.choice(ALL_COMPANIES)}\n  Amount: ${random.randint(5000, 200000):,}.00"
        img = Image.new("RGB", (800, 600), "#1a1a2e")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except OSError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            except OSError:
                font = ImageFont.load_default()
        y = 20
        for line in text.split("\n"):
            draw.text((20, y), line, fill="white", font=font)
            y += 24
        buf = BytesIO()
        img.save(buf, format="PNG")
        (screenshots_dir / f"banking_{i:03d}.png").write_bytes(buf.getvalue())
        count += 1
        progress.advance(task)

    # Crypto wallet screenshots
    for i in range(5):
        exchange = random.choice(CRYPTO_EXCHANGES)
        btc = round(random.uniform(0.5, 15.0), 4)
        eth = round(random.uniform(5, 200), 4)
        text = f"{exchange}\nCrypto Wallet\n\nBTC: {btc} (~${int(btc * 65000):,})\nETH: {eth} (~${int(eth * 3500):,})\nUSDT: {random.randint(10000, 500000):,}\n\nLast Transaction:\n  Sent {round(random.uniform(0.1, 5.0), 4)} BTC\n  To: {fake_en.sha256()[:16]}..."
        img = Image.new("RGB", (800, 600), "#0d1117")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except OSError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            except OSError:
                font = ImageFont.load_default()
        y = 20
        for line in text.split("\n"):
            draw.text((20, y), line, fill="#58a6ff", font=font)
            y += 24
        buf = BytesIO()
        img.save(buf, format="PNG")
        (screenshots_dir / f"crypto_{i:03d}.png").write_bytes(buf.getvalue())
        count += 1
        progress.advance(task)

    return count


def generate_business_records(base: Path, progress, task) -> int:
    """Generate contracts, meeting minutes, proposals in multiple languages."""
    biz_dir = base / "Documents" / "Business"
    biz_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # Contracts -- German, English, Spanish, Portuguese
    contract_locales = ["en", "de", "es", "pt", "en", "de", "en", "es", "pt", "en"]
    contract_headers = {
        "en": "CONSULTING SERVICES AGREEMENT",
        "de": "BERATUNGSDIENSTLEISTUNGSVERTRAG",
        "es": "CONTRATO DE SERVICIOS DE CONSULTORÍA",
        "pt": "CONTRATO DE PRESTAÇÃO DE SERVIÇOS DE CONSULTORIA",
    }
    contract_sections = {
        "en": ("SCOPE OF SERVICES", "COMPENSATION", "CONFIDENTIALITY", "GOVERNING LAW", "Client", "Consultant", "shall provide", "services including", "shall pay", "per month", "for a period of", "months", "Payment shall be made via wire transfer to", "Both parties agree to maintain strict confidentiality", "This agreement shall be governed by the laws of", "SIGNED"),
        "de": ("LEISTUNGSUMFANG", "VERGÜTUNG", "VERTRAULICHKEIT", "GELTENDES RECHT", "Auftraggeber", "Berater", "wird erbringen", "Dienstleistungen einschließlich", "wird zahlen", "pro Monat", "für einen Zeitraum von", "Monaten", "Die Zahlung erfolgt per Überweisung an", "Beide Parteien verpflichten sich zu strikter Vertraulichkeit", "Dieser Vertrag unterliegt dem Recht von", "UNTERSCHRIEBEN"),
        "es": ("ALCANCE DE LOS SERVICIOS", "COMPENSACIÓN", "CONFIDENCIALIDAD", "LEY APLICABLE", "Cliente", "Consultor", "proporcionará", "servicios incluyendo", "pagará", "por mes", "por un período de", "meses", "El pago se realizará por transferencia bancaria a", "Ambas partes acuerdan mantener estricta confidencialidad", "Este acuerdo se regirá por las leyes de", "FIRMADO"),
        "pt": ("ESCOPO DOS SERVIÇOS", "REMUNERAÇÃO", "CONFIDENCIALIDADE", "LEI APLICÁVEL", "Cliente", "Consultor", "fornecerá", "serviços incluindo", "pagará", "por mês", "por um período de", "meses", "O pagamento será feito por transferência bancária para", "Ambas as partes concordam em manter estrita confidencialidade", "Este contrato será regido pelas leis de", "ASSINADO"),
    }
    for i in range(10):
        locale = contract_locales[i]
        lf = LOCALE_FAKERS[locale]
        sec = contract_sections[locale]
        company_a = _locale_company(locale)
        company_b = random.choice(ALL_COMPANIES + [lf.company()])
        content = f"""{contract_headers[locale]}

Date: {_random_date().strftime('%B %d, %Y')}

BETWEEN:
  {company_a} ("{sec[4]}")
  {lf.address().replace(chr(10), ', ')}

AND:
  {company_b} ("{sec[5]}")
  {lf.address().replace(chr(10), ', ')}

1. {sec[0]}
   {sec[5]} {sec[6]} {sec[7]}:
   - {lf.sentence(nb_words=6)}
   - {lf.sentence(nb_words=5)}
   - {lf.sentence(nb_words=7)}

2. {sec[1]}
   {sec[4]} {sec[8]} ${random.randint(10000, 500000):,} {sec[9]}
   {sec[10]} {random.randint(6, 36)} {sec[11]}.
   
   {sec[12]}:
   Bank: {_locale_bank(locale)}
   Account: {lf.iban()}
   SWIFT: {lf.bothify('????##??###').upper()}

3. {sec[2]}
   {sec[13]}.

4. {sec[3]}
   {sec[14]} {lf.state() if hasattr(lf, 'state') else lf.city()}.

{sec[15]}:
_________________________          _________________________
{lf.name()}                      {lf.name()}
{company_a}                        {company_b}
"""
        (biz_dir / f"contract_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # Meeting minutes -- Japanese, English, Chinese, mixed
    minutes_locales = ["en", "ja", "zh", "en", "ja", "en", "zh", "en"]
    minutes_headers = {"en": "MEETING MINUTES", "ja": "会議議事録", "zh": "会议纪要"}
    for i in range(8):
        locale = minutes_locales[i]
        lf = LOCALE_FAKERS[locale]
        header = minutes_headers.get(locale, "MEETING MINUTES")
        attendees = [lf.name() for _ in range(random.randint(3, 6))]
        company = _locale_company(locale)
        content = f"""{header}
{company}
Date: {_random_date().strftime('%B %d, %Y')}
Location: {lf.city()}, {lf.country()}

Attendees: {', '.join(attendees)}

"""
        if locale == "ja":
            content += f"""議題:
1. 第{random.randint(1,4)}四半期の業績レビュー
2. {lf.sentence(nb_words=4)}に関する新イニシアチブ
3. {_locale_company('ja')}との提携状況

議事:

{attendees[0]}が会議を開き、四半期報告を発表しました。
{_locale_company('ja')}からの収益: ${random.randint(100000, 2000000):,}
営業費用: ${random.randint(50000, 500000):,}

{attendees[1]}がコンプライアンスの期限について懸念を示し、
監査期間前に{_locale_bank('ja')}の口座を再編することを提案しました。

{random.choice(attendees)}が${random.randint(50000, 1000000):,}を
{_locale_bank('ja')}のホールディング口座に「運営準備金」として移動することを提案しました。

アクション項目:
- {attendees[0]}: {_random_date().strftime('%B %d')}までに振込書類を準備
- {attendees[1]}: 法律顧問との面談を設定
"""
        elif locale == "zh":
            content += f"""议程:
1. 第{random.randint(1,4)}季度财务业绩回顾
2. 讨论新的{lf.sentence(nb_words=3)}计划
3. 更新{_locale_company('zh')}合作事项

纪要:

{attendees[0]}主持会议并报告了季度数据。
{_locale_company('zh')}收入: ${random.randint(100000, 2000000):,}
运营支出: ${random.randint(50000, 500000):,}

{attendees[1]}对合规时间表表示担忧，建议在审计期前
重组{_locale_bank('zh')}的账户。

{random.choice(attendees)}提议将${random.randint(50000, 1000000):,}转入
{_locale_bank('zh')}的控股账户作为"运营储备"。

行动项目:
- {attendees[0]}: 在{_random_date().strftime('%B %d')}前准备转账文件
- {attendees[1]}: 安排与法律顾问的会议
"""
        else:
            content += f"""AGENDA:
1. Review of Q{random.randint(1,4)} financial performance
2. Discussion of new {lf.sentence(nb_words=4)} initiative
3. Update on {random.choice(ALL_COMPANIES)} partnership

MINUTES:

{attendees[0]} opened the meeting and presented the quarterly figures.
Revenue from {random.choice(ALL_COMPANIES)}: ${random.randint(100000, 2000000):,}
Operating expenses: ${random.randint(50000, 500000):,}

{attendees[1]} raised concerns about the compliance timeline and suggested
restructuring the {random.choice(ALL_BANKS)} accounts before the audit period.

{random.choice(attendees)} proposed moving ${random.randint(50000, 1000000):,} to the
{random.choice(ALL_BANKS)} holding account for "operational reserves."

ACTION ITEMS:
- {attendees[0]}: Prepare transfer documentation by {_random_date().strftime('%B %d')}
- {attendees[1]}: Schedule meeting with legal counsel
"""
        (biz_dir / f"minutes_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # Business proposals -- Portuguese, English, mixed
    proposal_locales = ["en", "pt", "en", "pt", "en"]
    proposal_headers = {"en": "BUSINESS PROPOSAL", "pt": "PROPOSTA COMERCIAL"}
    for i in range(5):
        locale = proposal_locales[i]
        lf = LOCALE_FAKERS[locale]
        header = proposal_headers.get(locale, "BUSINESS PROPOSAL")
        company = _locale_company(locale)

        if locale == "pt":
            content = f"""{header}
{company}

Preparado para: {lf.name()}, {_locale_company('pt')}
Data: {_random_date().strftime('%B %d, %Y')}

RESUMO EXECUTIVO

{company} propõe fornecer serviços de {lf.sentence(nb_words=5)} com receita anual
projetada de ${random.randint(500000, 10000000):,}. O investimento inicial
necessário é de ${random.randint(100000, 2000000):,}.

ANÁLISE DE MERCADO

Mercados-alvo: {', '.join(lf.country() for _ in range(3))}
Crescimento projetado: {random.randint(15, 45)}% anualmente
Concorrentes-chave: {lf.company()}, {lf.company()}

PROJEÇÕES FINANCEIRAS

Ano 1: ${random.randint(200000, 2000000):,} receita
Ano 2: ${random.randint(500000, 5000000):,} receita
Ano 3: ${random.randint(1000000, 10000000):,} receita

Ponto de equilíbrio esperado em {random.randint(8, 24)} meses.

ESTRUTURA DE FINANCIAMENTO

Financiamento primário: {_locale_bank('pt')}
Secundário: Investidores privados via {_locale_company('pt')}
"""
        else:
            content = f"""{header}
{company}

Prepared for: {lf.name()}, {random.choice(ALL_COMPANIES)}
Date: {_random_date().strftime('%B %d, %Y')}

EXECUTIVE SUMMARY

{company} proposes to provide {lf.sentence(nb_words=5)} services with projected annual
revenue of ${random.randint(500000, 10000000):,}. The initial investment
required is ${random.randint(100000, 2000000):,}.

MARKET ANALYSIS

Target markets: {', '.join(lf.country() for _ in range(3))}
Projected growth: {random.randint(15, 45)}% annually
Key competitors: {lf.company()}, {lf.company()}

FINANCIAL PROJECTIONS

Year 1: ${random.randint(200000, 2000000):,} revenue
Year 2: ${random.randint(500000, 5000000):,} revenue
Year 3: ${random.randint(1000000, 10000000):,} revenue

Break-even expected in {random.randint(8, 24)} months.

FUNDING STRUCTURE

Primary funding: {random.choice(ALL_BANKS)}
Secondary: Private investors via {random.choice(ALL_COMPANIES)}
"""
        (biz_dir / f"proposal_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    return count


def generate_digital_artifacts(base: Path, progress, task) -> int:
    """Generate browser history, crypto wallets, notes in multiple languages."""
    digital_dir = base / "AppData"
    digital_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # Browser bookmarks
    bookmarks = {
        "bookmarks": [
            {"title": f"{random.choice(ALL_BANKS)} - Login", "url": f"https://{fake_en.domain_name()}/login"},
            *[{"title": f"{ex} - Exchange", "url": f"https://{ex}"} for ex in CRYPTO_EXCHANGES],
            {"title": "Offshore Company Formation", "url": f"https://{fake_en.domain_name()}/offshore-setup"},
            {"title": "FBAR Filing Requirements", "url": "https://www.irs.gov/businesses/small-businesses-self-employed/report-of-foreign-bank-and-financial-accounts-fbar"},
            {"title": "Anonymous LLC Formation", "url": f"https://{fake_en.domain_name()}/anonymous-llc"},
            {"title": "Gründung einer GmbH in Liechtenstein", "url": f"https://{fake_de.domain_name()}/gmbh-gruendung"},
            {"title": "Apertura de Sociedad Anónima - Panamá", "url": f"https://{fake_es.domain_name()}/sociedad-anonima"},
            {"title": "香港离岸公司注册", "url": f"https://{fake_zh.domain_name()}/offshore-hk"},
            *[{"title": fake.catch_phrase(), "url": fake_en.url()} for _ in range(8)],
        ]
    }
    (digital_dir / "bookmarks.json").write_text(
        json.dumps(bookmarks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    count += 1
    progress.advance(task)

    # Browser history
    history = []
    for _ in range(80):
        history.append({
            "url": random.choice([
                f"https://{random.choice(CRYPTO_EXCHANGES)}/trade",
                f"https://{fake_en.domain_name()}/banking",
                f"https://mail.{fake_en.domain_name()}/inbox",
                fake_en.url(),
                f"https://{random.choice(CRYPTO_EXCHANGES)}/wallet/withdraw",
            ]),
            "title": random.choice([
                "Trade BTC/USDT", "Wire Transfer - Confirm", fake_en.catch_phrase(),
                "Account Balance", "New Company Registration", "Property Listing",
                "Überweisungsbestätigung", "送金確認", "Confirmación de transferencia",
            ]),
            "visited_at": _random_date().isoformat(),
        })
    (digital_dir / "browser_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    count += 1
    progress.advance(task)

    # Crypto wallet exports
    for i in range(3):
        wallet = {
            "wallet_name": f"Wallet_{fake_en.lexify('???').upper()}",
            "addresses": [
                {"coin": "BTC", "address": f"bc1{fake_en.sha256()[:38]}", "balance": round(random.uniform(0.1, 10), 6)},
                {"coin": "ETH", "address": f"0x{fake_en.sha256()[:40]}", "balance": round(random.uniform(1, 100), 6)},
            ],
            "transactions": [
                {
                    "type": random.choice(["send", "receive"]),
                    "coin": random.choice(["BTC", "ETH", "USDT"]),
                    "amount": round(random.uniform(0.01, 5), 6),
                    "to": f"0x{fake_en.sha256()[:40]}" if random.random() > 0.5 else f"bc1{fake_en.sha256()[:38]}",
                    "timestamp": _random_date().isoformat(),
                    "tx_hash": fake_en.sha256(),
                }
                for _ in range(random.randint(10, 30))
            ],
        }
        (digital_dir / f"wallet_{i:03d}.json").write_text(json.dumps(wallet, indent=2), encoding="utf-8")
        count += 1
        progress.advance(task)

    # Personal notes -- multilingual, reflecting a suspect who operates in many languages
    notes_dir = base / "Documents" / "Notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    note_contents = [
        # English
        f"Account numbers:\n{random.choice(OFFSHORE_BANKS)}: {fake_en.iban()}\n{random.choice(OFFSHORE_BANKS)}: {fake_en.iban()}\n\nPIN: {fake_en.numerify('####')}\nOnline password: {fake_en.password()}",
        # English task list
        f"TODO:\n- Transfer ${random.randint(50000, 500000):,} from {random.choice(SHELL_COMPANIES)}\n- File {random.choice(SHELL_COMPANIES)} annual report\n- Meet with {fake_en.name()} about the property\n- Close {random.choice(OFFSHORE_BANKS)} account #{fake_en.numerify('######')}",
        # Mixed English/Spanish
        f"Meeting notes {_random_date().strftime('%m/%d')}:\n{fake_en.name()} dice que necesitamos mover los fondos de {random.choice(SHELL_COMPANIES)} antes de marzo.\nNueva estructura: {random.choice(SHELL_COMPANIES_INTL['es'])} -> {random.choice(SHELL_COMPANIES)} -> personal.\nHonorarios del abogado: ${random.randint(5000, 50000):,}",
        # German
        f"Notizen {_random_date().strftime('%d.%m.%Y')}:\nKontonummern:\n{random.choice(OFFSHORE_BANKS_INTL['de'])}: {fake_de.iban()}\n{random.choice(OFFSHORE_BANKS)}: {fake_de.iban()}\n\nÜberweisung an {random.choice(SHELL_COMPANIES_INTL['de'])}: ${random.randint(100000, 1000000):,}\nDringend vor der Steuerprüfung erledigen!",
        # Chinese
        f"备忘录 {_random_date().strftime('%Y/%m/%d')}:\n将 ${random.randint(100000, 500000):,} 从 {random.choice(SHELL_COMPANIES_INTL['zh'])} 转到 {random.choice(SHELL_COMPANIES)}\n联系 {fake_zh.name()} 关于房产交易\n{random.choice(OFFSHORE_BANKS_INTL.get('zh', OFFSHORE_BANKS))} 账户需要在月底前关闭",
        # Arabic
        f"ملاحظات {_random_date().strftime('%Y/%m/%d')}:\nتحويل ${random.randint(50000, 300000):,} من {random.choice(SHELL_COMPANIES_INTL['ar'])}\nالاتصال بالمحامي بخصوص {random.choice(SHELL_COMPANIES)}\n{random.choice(OFFSHORE_BANKS_INTL['ar'])} - الحساب يجب إغلاقه قبل نهاية الشهر",
        # Russian
        f"Заметки {_random_date().strftime('%d.%m.%Y')}:\nПеревести ${random.randint(50000, 500000):,} из {random.choice(SHELL_COMPANIES_INTL['ru'])}\nВстреча с {fake_ru.name()} по поводу недвижимости\n{random.choice(OFFSHORE_BANKS_INTL['ru'])} — счёт нужно закрыть",
        # Japanese
        f"メモ {_random_date().strftime('%Y/%m/%d')}:\n{random.choice(SHELL_COMPANIES_INTL['ja'])}から${random.randint(100000, 500000):,}を移動\n{fake_ja.name()}と不動産について面会\n{random.choice(ALL_BANKS)}の口座を月末までに閉鎖",
        # Crypto notes (English)
        f"Exchange rates checked {_random_date().strftime('%m/%d/%Y')}:\nBTC: ${random.randint(40000, 70000):,}\nETH: ${random.randint(2000, 4000):,}\nNeed to convert {round(random.uniform(1, 10), 2)} BTC to USD via {random.choice(CRYPTO_EXCHANGES)}",
        # Property notes (Portuguese)
        f"Propriedades para comprar:\n1. {fake_pt.address()} - R${random.randint(300000, 3000000):,}\n2. {fake_pt.address()} - R${random.randint(200000, 2000000):,}\nUsar {random.choice(SHELL_COMPANIES_INTL['pt'])} para compra.\nCartório: {fake_pt.company()}",
    ]
    for i, content in enumerate(note_contents):
        (notes_dir / f"note_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # SQLite database of client records
    db_path = digital_dir / "clients.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("""CREATE TABLE clients (
        id INTEGER PRIMARY KEY, name TEXT, company TEXT, email TEXT,
        phone TEXT, account_number TEXT, total_managed REAL, status TEXT, country TEXT
    )""")
    for j in range(50):
        locale = _pick_locale()
        lf = LOCALE_FAKERS[locale]
        c.execute(
            "INSERT INTO clients VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                j + 1,
                lf.name(),
                random.choice(ALL_COMPANIES + [lf.company()]),
                lf.email(),
                lf.phone_number(),
                lf.iban(),
                round(random.uniform(10000, 5000000), 2),
                random.choice(["Active", "Active", "Active", "Dormant", "Closed"]),
                lf.country(),
            ),
        )
    conn.commit()
    conn.close()
    count += 1
    progress.advance(task)

    return count


# ---------------------------------------------------------------------------
# Evidence threads: pre-scripted cross-document investigative storylines
# ---------------------------------------------------------------------------
# Each thread uses FIXED identifiers so the same people, amounts, references,
# and dates appear across multiple file types -- exactly how real laundering
# operations leave trails across a hard drive.

THREAD_OCEANIC = {
    "name": "Operation Oceanic",
    "suspect": "Victor Reyes",
    "accomplice": "Elena Marchetti",
    "accomplice_role": "attorney",
    "company_a": "Crescent Bay Enterprises",
    "company_b": "Oceanic Resources Group",
    "amount": 2_450_000,
    "currency": "USD",
    "ref": "TXN-7734-CB89",
    "property_address": "1847 Coral Ridge Drive, Miami FL 33139",
    "bank": "Cayman National Bank",
    "account_tail": "8891",
    "iban": "KY21CAYB0001000012348891",
    "swift": "CAYB KY KI",
    "timeline_start": datetime(2024, 3, 10),
    "timeline_end": datetime(2024, 6, 15),
    "suspect_email": "v.reyes@crescent-bay.com",
    "accomplice_email": "elena.marchetti@lawfirm-intl.com",
    "agent_email": "listings@suncoast-realty.com",
}

THREAD_ALPINE = {
    "name": "Operation Alpine",
    "suspect": "Klaus Brenner",
    "contact": "Hiroshi Tanaka",
    "company_de": "Rheinische Handelsgesellschaft mbH",
    "company_ch": "Nordkap Finanz AG",
    "company_jp": "太陽商事株式会社",
    "amount": 875_000,
    "currency": "EUR",
    "ref": "INV-4419-ZH23",
    "bank": "Zürcher Privatbank AG",
    "account_tail": "5537",
    "iban": "CH9300762011623852955537",
    "swift": "ZRCHCH22",
    "timeline_start": datetime(2023, 9, 5),
    "timeline_end": datetime(2023, 11, 20),
    "suspect_email": "k.brenner@rheinische-hg.de",
    "contact_email": "h.tanaka@taiyo-shoji.co.jp",
    "nordkap_email": "finanzen@nordkap-ag.ch",
}

THREAD_SILK = {
    "name": "Operation Silk Road",
    "suspect": "Wei Chen",
    "contact": "Khalid Al-Rashidi",
    "company_hk": "金桥投资管理集团",
    "company_dubai": "شركة الخليج للاستثمار المحدودة",
    "btc_amount": 12.75,
    "btc_price": 65_000,
    "fiat_amount": 828_750,
    "currency": "USD",
    "ref": "TXN-9021-GQ47",
    "wallet_address": "bc1q7x9k4m2p3f8n5v6j1w0r2t8y4a6s3kf2",
    "tx_hash": "a4f3e8b91c7d2a0f5e6b3c8d7a9f1e2b4c6d8a0f3e5b7c9d1a2f4e6b8c0d2a4",
    "bank": "بنك الخليج الدولي",
    "account_tail": "3306",
    "timeline_start": datetime(2025, 1, 12),
    "timeline_end": datetime(2025, 3, 8),
    "suspect_email": "wei.chen@jinqiao-invest.hk",
    "contact_email": "khalid@gulf-invest.ae",
}


def _thread_date(t: dict, day_offset: int = 0) -> datetime:
    """Return a date within a thread's timeline."""
    base = t["timeline_start"]
    span = (t["timeline_end"] - base).days
    offset = min(day_offset, span)
    return base + timedelta(days=offset)


def _generate_oceanic(base: Path, progress, task) -> int:
    """Operation Oceanic: real estate money laundering via shell companies."""
    t = THREAD_OCEANIC
    count = 0

    # --- Email 1: Elena -> Victor confirming wire ---
    email_dir = base / "Communications" / "Email"
    email_dir.mkdir(parents=True, exist_ok=True)
    msg = EmailMessage()
    msg["Subject"] = f"Re: Wire transfer confirmation - {t['ref']}"
    msg["From"] = t["accomplice_email"]
    msg["To"] = t["suspect_email"]
    msg["Date"] = _thread_date(t, 15).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg.set_content(
        f"Victor,\n\n"
        f"The wire transfer of ${t['amount']:,} from {t['company_a']} to {t['company_b']} "
        f"has been completed via SWIFT.\n\n"
        f"Reference: {t['ref']}\n"
        f"Receiving bank: {t['bank']}, account ending {t['account_tail']}\n"
        f"IBAN: {t['iban']}\n\n"
        f"The funds should clear within 2-3 business days. I've structured this as "
        f"consulting fees per our earlier discussion.\n\n"
        f"Please confirm once you see the credit on your end.\n\n"
        f"Best regards,\nElena Marchetti\nInternational Legal Counsel"
    )
    (email_dir / "re_wire_transfer_confirmation.eml").write_text(msg.as_string(), encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Email 2: Real estate agent confirming closing ---
    msg2 = EmailMessage()
    msg2["Subject"] = f"Closing confirmed - {t['property_address']}"
    msg2["From"] = t["agent_email"]
    msg2["To"] = t["suspect_email"]
    msg2["Cc"] = t["accomplice_email"]
    msg2["Date"] = _thread_date(t, 85).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg2.set_content(
        f"Dear Mr. Reyes,\n\n"
        f"I'm pleased to confirm that the closing on the property at "
        f"{t['property_address']} has been completed successfully.\n\n"
        f"Purchase price: ${t['amount']:,}\n"
        f"Buyer: {t['company_b']}\n"
        f"Title has been transferred and recorded with Miami-Dade County.\n\n"
        f"Congratulations on your new property!\n\n"
        f"Best,\nSuncoast Realty Group"
    )
    (email_dir / "closing_confirmed_coral_ridge.eml").write_text(msg2.as_string(), encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Transaction ledger CSV ---
    fin_dir = base / "Documents" / "Financial"
    fin_dir.mkdir(parents=True, exist_ok=True)
    path = fin_dir / "transactions_Q2_2024.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "From", "To", "Amount", "Currency", "Reference", "Type", "Locale"])
        writer.writerow([
            _thread_date(t, 15).strftime("%Y-%m-%d"),
            t["company_a"], t["company_b"], t["amount"], "USD", t["ref"], "Wire Transfer", "en",
        ])
        writer.writerow([
            _thread_date(t, 18).strftime("%Y-%m-%d"),
            t["company_b"], "Suncoast Title & Escrow LLC", t["amount"], "USD",
            "TXN-7735-CB90", "Wire Transfer", "en",
        ])
    count += 1; progress.advance(task)

    # --- Bank statement showing the credit ---
    path = fin_dir / f"statement_{t['bank'].replace(' ', '_')}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Description", "Debit", "Credit", "Balance"])
        balance = 347_219.50
        writer.writerow([_thread_date(t, 10).strftime("%Y-%m-%d"), "Opening balance", "", "", f"{balance:.2f}"])
        balance += t["amount"]
        writer.writerow([
            _thread_date(t, 15).strftime("%Y-%m-%d"),
            f"SWIFT Credit - {t['company_a']} - Ref {t['ref']}",
            "", f"{t['amount']:.2f}", f"{balance:.2f}",
        ])
        balance -= t["amount"]
        writer.writerow([
            _thread_date(t, 18).strftime("%Y-%m-%d"),
            f"Wire Out - Suncoast Title & Escrow - Property Closing",
            f"{t['amount']:.2f}", "", f"{balance:.2f}",
        ])
        writer.writerow([
            _thread_date(t, 25).strftime("%Y-%m-%d"),
            "Monthly maintenance fee", "25.00", "", f"{balance - 25:.2f}",
        ])
    count += 1; progress.advance(task)

    # --- Consulting contract ---
    biz_dir = base / "Documents" / "Business"
    biz_dir.mkdir(parents=True, exist_ok=True)
    contract = f"""CONSULTING SERVICES AGREEMENT

Date: {_thread_date(t, 5).strftime('%B %d, %Y')}

BETWEEN:
  {t['company_a']} ("Client")
  2200 Biscayne Boulevard, Suite 1450, Miami FL 33137

AND:
  {t['company_b']} ("Consultant")
  P.O. Box 1288, George Town, Grand Cayman KY1-1108

1. SCOPE OF SERVICES
   Consultant shall provide strategic advisory and asset management services
   including but not limited to:
   - Real estate acquisition due diligence
   - Portfolio diversification analysis
   - International asset structuring

2. COMPENSATION
   Client shall pay Consultant a one-time fee of ${t['amount']:,} USD
   upon execution of this agreement.

   Payment shall be made via wire transfer to:
   Bank: {t['bank']}
   IBAN: {t['iban']}
   SWIFT: {t['swift']}
   Reference: {t['ref']}

3. CONFIDENTIALITY
   Both parties agree to maintain strict confidentiality regarding all financial
   arrangements, client identities, and operational details. No disclosure to
   third parties without written consent.

4. GOVERNING LAW
   This agreement shall be governed by the laws of the Cayman Islands.

SIGNED:
_________________________          _________________________
{t['suspect']}                     Representative
{t['company_a']}                   {t['company_b']}
"""
    (biz_dir / "consulting_services_agreement.txt").write_text(contract, encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Property photo ---
    media_dir = base / "Media" / "Photos"
    media_dir.mkdir(parents=True, exist_ok=True)
    prop_text = (
        f"PROPERTY LISTING\n\n"
        f"{t['property_address']}\n\n"
        f"Luxury Beachfront Condominium\n"
        f"3 Bed / 3.5 Bath / 2,847 sq ft\n\n"
        f"Asking Price: ${t['amount']:,}\n"
        f"Buyer: {t['company_b']}\n"
        f"Closing Date: {_thread_date(t, 85).strftime('%B %d, %Y')}"
    )
    img_bytes = _make_image(prop_text, 800, 600, bg="#f5f0e0")
    (media_dir / "coral_ridge_listing.jpg").write_bytes(img_bytes)
    count += 1; progress.advance(task)

    # --- Personal note ---
    notes_dir = base / "Documents" / "Notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    note = (
        f"Coral Ridge closing {_thread_date(t, 85).strftime('%B %d')}.\n"
        f"Elena handling Cayman transfer - {t['ref']}\n"
        f"${t['amount']:,} from Crescent Bay -> Oceanic -> title company\n"
        f"Account ****{t['account_tail']} at {t['bank']}\n"
        f"IBAN: {t['iban']}\n\n"
        f"Make sure consulting agreement is dated BEFORE the wire.\n"
        f"Elena says structure is clean. Destroy this note after closing."
    )
    (notes_dir / "closing_notes.txt").write_text(note, encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Chat log ---
    chat_dir = base / "Communications" / "ChatLogs"
    chat_dir.mkdir(parents=True, exist_ok=True)
    chat = [
        {"timestamp": _thread_date(t, 10).isoformat(), "sender": "Victor", "phone": "+1-305-555-0147",
         "message": "Elena, is the Cayman account ready? Need to move the 2.45M this week."},
        {"timestamp": _thread_date(t, 10).replace(hour=14, minute=22).isoformat(), "sender": "Elena", "phone": "+41-78-555-0933",
         "message": "Account 8891 is active. I'll initiate the SWIFT from Crescent Bay tomorrow."},
        {"timestamp": _thread_date(t, 11).isoformat(), "sender": "Elena", "phone": "+41-78-555-0933",
         "message": f"Wire sent. Reference {t['ref']}. Should clear by Friday."},
        {"timestamp": _thread_date(t, 14).isoformat(), "sender": "Victor", "phone": "+1-305-555-0147",
         "message": "Funds landed. Moving to escrow for Coral Ridge next week."},
        {"timestamp": _thread_date(t, 14).replace(hour=16).isoformat(), "sender": "Elena", "phone": "+41-78-555-0933",
         "message": "Good. Make sure the consulting agreement is in the file before the title search."},
        {"timestamp": _thread_date(t, 80).isoformat(), "sender": "Victor", "phone": "+1-305-555-0147",
         "message": "Closing confirmed for the 15th. Suncoast has everything they need."},
        {"timestamp": _thread_date(t, 80).replace(hour=19).isoformat(), "sender": "Elena", "phone": "+41-78-555-0933",
         "message": "Perfect. Delete this thread after closing. Use the new number for future contact."},
    ]
    (chat_dir / "chat_export_007.json").write_text(
        json.dumps(chat, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    return count


def _generate_alpine(base: Path, progress, task) -> int:
    """Operation Alpine: cross-border trade-based laundering (German/Swiss/Japanese)."""
    t = THREAD_ALPINE
    count = 0

    # --- German invoice ---
    inv_dir = base / "Documents" / "Financial" / "Invoices"
    inv_dir.mkdir(parents=True, exist_ok=True)
    invoice = f"""RECHNUNG

Rechnung Nr.: {t['ref']}
Datum: {_thread_date(t, 12).strftime('%d. %B %Y')}

Von: {t['company_de']}
     Rheinufer 42, 40213 Düsseldorf, Deutschland

An:  {t['company_ch']}
     Bahnhofstrasse 15, 8001 Zürich, Schweiz

{'Artikel':<40} {'Menge':>5} {'Stückpreis':>12} {'Gesamt':>12}
{'-' * 75}
{'Industrieberatung Q3 2023':<40} {'1':>5} {'625,000.00':>12} {'625,000.00':>12}
{'Marktanalyse Asien-Pazifik':<40} {'1':>5} {'175,000.00':>12} {'175,000.00':>12}
{'Logistikkoordination Tokyo':<40} {'1':>5} {'75,000.00':>12} {'75,000.00':>12}
{'-' * 75}
{'GESAMT':>57} {'875,000.00':>12} EUR

Zahlungsbedingungen: Sofort fällig
Bank: {t['bank']}
IBAN: {t['iban']}
SWIFT: {t['swift']}
Referenz: {t['ref']}
"""
    (inv_dir / f"{t['ref']}.txt").write_text(invoice, encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Email (German): Klaus confirming SWIFT transfer ---
    email_dir = base / "Communications" / "Email"
    email_dir.mkdir(parents=True, exist_ok=True)
    msg = EmailMessage()
    msg["Subject"] = f"AW: Überweisungsbestätigung - {t['ref']}"
    msg["From"] = t["suspect_email"]
    msg["To"] = t["nordkap_email"]
    msg["Date"] = _thread_date(t, 14).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg["X-Language"] = "de"
    msg.set_content(
        f"Sehr geehrte Damen und Herren,\n\n"
        f"hiermit bestätige ich die SWIFT-Überweisung von EUR {t['amount']:,} "
        f"an {t['company_ch']}.\n\n"
        f"IBAN: {t['iban']}\n"
        f"SWIFT: {t['swift']}\n"
        f"Referenz: {t['ref']}\n\n"
        f"Die Zahlung betrifft die Rechnung für Industrieberatung Q3 2023. "
        f"Bitte bestätigen Sie den Eingang.\n\n"
        f"Mit freundlichen Grüßen,\n{t['suspect']}\n{t['company_de']}"
    )
    (email_dir / "aw_ueberweisungsbestaetigung.eml").write_text(msg.as_string(), encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Email (Japanese): Hiroshi noting goods don't match invoice ---
    msg2 = EmailMessage()
    msg2["Subject"] = "Re: 送金確認 - チューリッヒ手配"
    msg2["From"] = t["contact_email"]
    msg2["To"] = t["suspect_email"]
    msg2["Date"] = _thread_date(t, 30).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg2["X-Language"] = "ja"
    msg2.set_content(
        f"Brenner様\n\n"
        f"{t['company_jp']}の田中です。\n\n"
        f"チューリッヒからの送金EUR {t['amount']:,}を確認しました。"
        f"ただし、実際に受け取った商品は請求書の金額と一致しません。"
        f"「Industrieberatung」の項目は実際のサービスとは異なるようです。\n\n"
        f"次回の取引については、より現実的な品目説明を使用することをお勧めします。"
        f"東京側の書類は整っています。\n\n"
        f"よろしくお願いします。\n田中 博\n{t['company_jp']}"
    )
    (email_dir / "re_soukin_kakunin.eml").write_text(msg2.as_string(), encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Meeting minutes (Japanese) ---
    biz_dir = base / "Documents" / "Business"
    biz_dir.mkdir(parents=True, exist_ok=True)
    minutes = f"""会議議事録
{t['company_jp']}
Date: {_thread_date(t, 35).strftime('%Y年%m月%d日')}
Location: 東京都港区, 日本

出席者: 田中 博, 山田 太郎, 佐藤 花子

議題:
1. チューリッヒ手配の状況確認
2. {t['company_de']}との取引レビュー
3. 今後の資金フロー計画

議事:

田中がチューリッヒからのEUR {t['amount']:,}の受領を報告しました。
{t['bank']}経由、参照番号{t['ref']}。

山田は請求書の品目説明について懸念を表明しました。
「Industrieberatung」（産業コンサルティング）は実際の取引内容と
一致しないため、今後はより適切な説明を使用する必要があります。

佐藤は{t['company_ch']}を経由する構造を維持することを提案し、
次回の送金は来四半期に予定されていると報告しました。

アクション項目:
- 田中: Brenner氏に請求書の修正を依頼
- 山田: {t['company_jp']}の帳簿を更新
- 佐藤: 次回送金のためのチャネルを準備
"""
    (biz_dir / "meeting_minutes_Q3.txt").write_text(minutes, encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Bank screenshot (Zurich) ---
    screenshots_dir = base / "Media" / "Screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    bank_text = (
        f"{t['bank']}\n"
        f"Online Banking\n\n"
        f"Konto: ****{t['account_tail']}\n"
        f"IBAN: {t['iban']}\n\n"
        f"Letzte Transaktionen:\n\n"
        f"  {_thread_date(t, 14).strftime('%d.%m.%Y')}  EINGANG\n"
        f"  Von: {t['company_de']}\n"
        f"  Referenz: {t['ref']}\n"
        f"  Betrag: EUR {t['amount']:,}.00\n\n"
        f"  {_thread_date(t, 16).strftime('%d.%m.%Y')}  AUSGANG\n"
        f"  An: {t['company_jp']}\n"
        f"  Betrag: EUR {t['amount']:,}.00"
    )
    img = Image.new("RGB", (800, 600), "#1a1a2e")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except OSError:
            font = ImageFont.load_default()
    y = 20
    for line in bank_text.split("\n"):
        draw.text((20, y), line, fill="white", font=font)
        y += 22
    buf = BytesIO()
    img.save(buf, format="PNG")
    (screenshots_dir / "zurich_banking_screenshot.png").write_bytes(buf.getvalue())
    count += 1; progress.advance(task)

    # --- Transaction CSV ---
    fin_dir = base / "Documents" / "Financial"
    fin_dir.mkdir(parents=True, exist_ok=True)
    path = fin_dir / "transactions_Q3_2023.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "From", "To", "Amount", "Currency", "Reference", "Type", "Locale"])
        writer.writerow([
            _thread_date(t, 14).strftime("%Y-%m-%d"),
            t["company_de"], t["company_ch"], t["amount"], "EUR", t["ref"], "SWIFT", "de",
        ])
        writer.writerow([
            _thread_date(t, 16).strftime("%Y-%m-%d"),
            t["company_ch"], t["company_jp"], t["amount"], "EUR",
            "TXN-4420-ZH24", "SWIFT", "ja",
        ])
    count += 1; progress.advance(task)

    # --- Personal note (German) ---
    notes_dir = base / "Documents" / "Notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    note = (
        f"Notizen - Zürich Arrangement\n\n"
        f"{t['bank']}\n"
        f"IBAN: {t['iban']}\n"
        f"SWIFT: {t['swift']}\n"
        f"Konto: ****{t['account_tail']}\n\n"
        f"Nächste Überweisung: EUR {t['amount']:,}\n"
        f"Referenz: {t['ref']}\n"
        f"{t['company_de']} -> {t['company_ch']} -> {t['company_jp']}\n\n"
        f"Tanaka bestätigt Empfang in Tokyo.\n"
        f"Rechnung muss VOR der Überweisung datiert sein!\n"
        f"Nächste Tranche: Q1 2024"
    )
    (notes_dir / "zurich_notes.txt").write_text(note, encoding="utf-8")
    count += 1; progress.advance(task)

    return count


def _generate_silk(base: Path, progress, task) -> int:
    """Operation Silk Road: crypto-to-fiat conversion (Chinese/Arabic)."""
    t = THREAD_SILK
    count = 0

    # --- Crypto wallet export ---
    digital_dir = base / "AppData"
    digital_dir.mkdir(parents=True, exist_ok=True)
    wallet = {
        "wallet_name": "main_wallet",
        "addresses": [
            {"coin": "BTC", "address": t["wallet_address"], "balance": 0.0312},
            {"coin": "ETH", "address": "0x3a7f2b9c1d4e8f6a5b0c9d2e7f1a3b5c8d4e6f0a", "balance": 45.2310},
        ],
        "transactions": [
            {
                "type": "send", "coin": "BTC", "amount": t["btc_amount"],
                "to": "bc1qm5r8v3n7k2p4j6h9w0x1y3z5a8b2c4d6f0g",
                "timestamp": _thread_date(t, 8).isoformat(),
                "tx_hash": t["tx_hash"],
                "note": f"Conversion - {t['ref']}",
            },
            {
                "type": "receive", "coin": "BTC", "amount": 15.0,
                "to": t["wallet_address"],
                "timestamp": _thread_date(t, 1).isoformat(),
                "tx_hash": "b7c2d4e6f8a0b1c3d5e7f9a2b4c6d8e0f1a3b5c7d9e0f2a4b6c8d0e2f4a6b8",
            },
            {
                "type": "send", "coin": "USDT", "amount": 50000.0,
                "to": "0x9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e",
                "timestamp": _thread_date(t, 15).isoformat(),
                "tx_hash": "c8d0e2f4a6b8c1d3e5f7a9b2c4d6e8f0a1b3c5d7e9f0a2b4c6d8e0f2a4b6c8",
            },
        ],
    }
    (digital_dir / "wallet_export.json").write_text(
        json.dumps(wallet, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Chat log (Chinese): Wei discussing the conversion ---
    chat_dir = base / "Communications" / "ChatLogs"
    chat_dir.mkdir(parents=True, exist_ok=True)
    chat_zh = [
        {"timestamp": _thread_date(t, 5).isoformat(), "sender": "Wei", "phone": "+852-5555-0821",
         "message": "12.75个比特币准备好了。按65000美元计算，总共828,750。", "language": "zh"},
        {"timestamp": _thread_date(t, 5).replace(hour=15).isoformat(), "sender": "李明", "phone": "+852-5555-0934",
         "message": "金桥那边账户已经准备好。什么时候发送？", "language": "zh"},
        {"timestamp": _thread_date(t, 6).isoformat(), "sender": "Wei", "phone": "+852-5555-0821",
         "message": "明天通过Binance转出。地址是bc1q7x9开头的那个。", "language": "zh"},
        {"timestamp": _thread_date(t, 8).isoformat(), "sender": "Wei", "phone": "+852-5555-0821",
         "message": f"已发送。交易哈希：{t['tx_hash'][:16]}... 等迪拜那边确认。", "language": "zh"},
        {"timestamp": _thread_date(t, 9).isoformat(), "sender": "李明", "phone": "+852-5555-0934",
         "message": "Khalid确认收到。参考编号TXN-9021-GQ47。款项将在本周内到达迪拜账户。", "language": "zh"},
    ]
    (chat_dir / "chat_export_008.json").write_text(
        json.dumps(chat_zh, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Chat log (Arabic): Khalid confirming receipt ---
    chat_ar = [
        {"timestamp": _thread_date(t, 8).replace(hour=18).isoformat(), "sender": "Khalid", "phone": "+971-55-555-0412",
         "message": f"تم استلام البيتكوين. 12.75 BTC. المرجع: {t['ref']}", "language": "ar"},
        {"timestamp": _thread_date(t, 9).isoformat(), "sender": "أحمد", "phone": "+971-55-555-0688",
         "message": "سيتم تحويل المبلغ $828,750 إلى حساب شركة الخليج خلال 48 ساعة.", "language": "ar"},
        {"timestamp": _thread_date(t, 10).isoformat(), "sender": "Khalid", "phone": "+971-55-555-0412",
         "message": f"تم التحويل. {t['bank']} - الحساب ****{t['account_tail']}. أبلغ وي تشن.", "language": "ar"},
        {"timestamp": _thread_date(t, 11).isoformat(), "sender": "Khalid", "phone": "+971-55-555-0412",
         "message": "احذف هذه المحادثة. استخدم القناة الجديدة للتواصل القادم.", "language": "ar"},
    ]
    (chat_dir / "chat_export_009.json").write_text(
        json.dumps(chat_ar, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Email (Chinese): Wei to 金桥 discussing conversion ---
    email_dir = base / "Communications" / "Email"
    email_dir.mkdir(parents=True, exist_ok=True)
    msg = EmailMessage()
    msg["Subject"] = f"回复：BTC转换计划 - {t['ref']}"
    msg["From"] = t["suspect_email"]
    msg["To"] = "operations@jinqiao-invest.hk"
    msg["Date"] = _thread_date(t, 4).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg["X-Language"] = "zh"
    msg.set_content(
        f"操作团队，\n\n"
        f"请准备以下转换：\n\n"
        f"数量：{t['btc_amount']} BTC\n"
        f"当前价格：${t['btc_price']:,}/BTC\n"
        f"预计法币金额：${t['fiat_amount']:,}\n\n"
        f"路径：{t['company_hk']} -> {t['company_dubai']}\n"
        f"参考编号：{t['ref']}\n\n"
        f"迪拜方面的联系人Khalid已确认接收账户就绪。\n"
        f"请在本周内完成Binance提款。\n\n"
        f"陈伟"
    )
    (email_dir / "re_btc_conversion.eml").write_text(msg.as_string(), encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Bank statement (Arabic bank) ---
    fin_dir = base / "Documents" / "Financial"
    fin_dir.mkdir(parents=True, exist_ok=True)
    path = fin_dir / "statement_gulf_international.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Description", "Debit", "Credit", "Balance"])
        balance = 125_430.00
        writer.writerow([_thread_date(t, 1).strftime("%Y-%m-%d"), "Opening balance", "", "", f"{balance:.2f}"])
        balance += t["fiat_amount"]
        writer.writerow([
            _thread_date(t, 10).strftime("%Y-%m-%d"),
            f"Credit - {t['company_hk']} - BTC Conversion - Ref {t['ref']}",
            "", f"{t['fiat_amount']:.2f}", f"{balance:.2f}",
        ])
        writer.writerow([
            _thread_date(t, 18).strftime("%Y-%m-%d"),
            f"Transfer to {t['company_dubai']} operating account",
            f"{500_000:.2f}", "", f"{balance - 500_000:.2f}",
        ])
    count += 1; progress.advance(task)

    # --- Transaction CSV ---
    path = fin_dir / "transactions_Q1_2025.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "From", "To", "Amount", "Currency", "Reference", "Type", "Locale"])
        writer.writerow([
            _thread_date(t, 10).strftime("%Y-%m-%d"),
            t["company_hk"], t["company_dubai"], t["fiat_amount"], "USD", t["ref"], "Wire Transfer", "zh",
        ])
    count += 1; progress.advance(task)

    # --- Note (mixed Chinese/English) ---
    notes_dir = base / "Documents" / "Notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    note = (
        f"BTC conversion计划:\n\n"
        f"BTC price: ${t['btc_price']:,}\n"
        f"Amount: {t['btc_amount']} BTC = ${t['fiat_amount']:,}\n\n"
        f"钱包地址: {t['wallet_address']}\n"
        f"交易哈希: {t['tx_hash'][:32]}...\n\n"
        f"路径: {t['company_hk']} -> {t['company_dubai']}\n"
        f"Khalid confirmed. Dubai account ****{t['account_tail']} ready.\n"
        f"{t['bank']}\n\n"
        f"参考编号: {t['ref']}\n"
        f"完成后删除此备忘录。"
    )
    (notes_dir / "conversion_memo.txt").write_text(note, encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Browser history entries ---
    history_path = base / "AppData" / "browser_history_export.json"
    history = [
        {"url": "https://binance.com/wallet/withdraw", "title": "Withdraw BTC - Binance",
         "visited_at": _thread_date(t, 7).isoformat()},
        {"url": "https://binance.com/wallet/withdraw", "title": "Withdraw BTC - Binance",
         "visited_at": _thread_date(t, 8).isoformat()},
        {"url": "https://blockchain.com/explorer", "title": "BTC Transaction Tracker",
         "visited_at": _thread_date(t, 8).replace(hour=15).isoformat()},
        {"url": f"https://blockchain.com/btc/tx/{t['tx_hash'][:16]}", "title": "TX Confirmation",
         "visited_at": _thread_date(t, 9).isoformat()},
        {"url": "https://coinmarketcap.com/currencies/bitcoin/", "title": "Bitcoin Price - CoinMarketCap",
         "visited_at": _thread_date(t, 6).isoformat()},
        {"url": "https://coinmarketcap.com/currencies/bitcoin/", "title": "Bitcoin Price - CoinMarketCap",
         "visited_at": _thread_date(t, 7).isoformat()},
    ]
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    count += 1; progress.advance(task)

    return count


def generate_evidence_threads(base: Path, progress, task) -> int:
    """Generate cross-document evidence threads for investigative storylines.

    Creates 3 pre-scripted multi-document operations where fixed identifiers
    (names, companies, amounts, references) appear across emails, spreadsheets,
    bank statements, contracts, images, chat logs, and notes.
    """
    count = 0
    count += _generate_oceanic(base, progress, task)
    count += _generate_alpine(base, progress, task)
    count += _generate_silk(base, progress, task)
    return count


# ---------------------------------------------------------------------------
# Workshop scenario: sanctions evasion via trade-based fraud
# ---------------------------------------------------------------------------
# Single investigative thread for a timed workshop. Internal constant names
# are never used in filenames or document content.

THREAD_PHANTOM_FREIGHT = {
    "suspect": "David Okafor",
    "broker": "Nadia Petrov",
    "intermediary": "Rashid Al-Mansoori",
    "company_lagos": "TransGlobal Freight Services Ltd",
    "company_odessa": "Black Sea Maritime Logistics",
    "company_dubai": "GulfStar Trading FZE",
    "amount_1": 340_000,
    "amount_2": 275_000,
    "currency": "USD",
    "ref_1": "SHP-2281-PF",
    "ref_2": "SHP-2282-PF",
    "bank_dubai": "Emirates NBD",
    "account_tail": "4471",
    "iban_dubai": "AE070331234567890124471",
    "swift_dubai": "EABORAE1XXX",
    "bank_odessa": "ПриватБанк",
    "iban_odessa": "UA903052992990004149123456789",
    "timeline_start": datetime(2024, 8, 5),
    "timeline_end": datetime(2024, 11, 20),
    "suspect_email": "d.okafor@transglobal-freight.ng",
    "broker_email": "n.petrov@blacksea-shipping.ua",
    "intermediary_email": "r.almansoori@gulfstar-trading.ae",
    "compliance_email": "compliance@transglobal-freight.ng",
}


def _generate_phantom_freight(base: Path, progress, task) -> int:
    """Workshop scenario: sanctions evasion via shell shipping invoices."""
    t = THREAD_PHANTOM_FREIGHT
    count = 0

    # --- Email 1: Nadia -> David confirming shipment booking ---
    email_dir = base / "Communications" / "Email"
    email_dir.mkdir(parents=True, exist_ok=True)
    msg = EmailMessage()
    msg["Subject"] = f"Re: Shipment booking confirmation - {t['ref_1']}"
    msg["From"] = t["broker_email"]
    msg["To"] = t["suspect_email"]
    msg["Date"] = _thread_date(t, 12).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg.set_content(
        f"David,\n\n"
        f"Booking confirmed for the industrial equipment shipment.\n\n"
        f"Reference: {t['ref_1']}\n"
        f"Route: Lagos -> Odessa -> Dubai (via transshipment)\n"
        f"Declared value: ${t['amount_1']:,}\n"
        f"Vessel: MV Northern Promise, ETD {_thread_date(t, 20).strftime('%d %b %Y')}\n\n"
        f"Payment should be wired to the Dubai account before loading.\n"
        f"I'll send the bill of lading once the container is sealed.\n\n"
        f"Regards,\nNadia"
    )
    (email_dir / "re_shipment_booking_confirmation.eml").write_text(
        msg.as_string(), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Email 2: Payment instructions from intermediary ---
    msg2 = EmailMessage()
    msg2["Subject"] = "Payment instructions - freight forwarding"
    msg2["From"] = t["intermediary_email"]
    msg2["To"] = t["suspect_email"]
    msg2["Cc"] = t["broker_email"]
    msg2["Date"] = _thread_date(t, 14).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg2.set_content(
        f"Mr. Okafor,\n\n"
        f"Please arrange payment for the freight forwarding services as follows:\n\n"
        f"Beneficiary: {t['company_dubai']}\n"
        f"Bank: {t['bank_dubai']}\n"
        f"IBAN: {t['iban_dubai']}\n"
        f"SWIFT: {t['swift_dubai']}\n"
        f"Amount: ${t['amount_1']:,} USD\n"
        f"Reference: {t['ref_1']}\n\n"
        f"The second shipment ({t['ref_2']}) for ${t['amount_2']:,} should follow "
        f"within 30 days using the same routing.\n\n"
        f"Best regards,\nR. Al-Mansoori\n{t['company_dubai']}"
    )
    (email_dir / "payment_instructions_freight.eml").write_text(
        msg2.as_string(), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Email 3: Internal compliance warning (ignored) ---
    msg3 = EmailMessage()
    msg3["Subject"] = "URGENT: Sanctions screening flag - GulfStar Trading"
    msg3["From"] = t["compliance_email"]
    msg3["To"] = t["suspect_email"]
    msg3["Date"] = _thread_date(t, 18).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg3.set_content(
        f"David,\n\n"
        f"Our automated sanctions screening flagged GulfStar Trading FZE "
        f"as a potential match against the OFAC SDN list. The beneficial owner "
        f"appears on a secondary watchlist.\n\n"
        f"Please do NOT proceed with any payments to this entity until "
        f"we complete enhanced due diligence.\n\n"
        f"I need the following documentation:\n"
        f"- End-user certificate for the equipment\n"
        f"- Customs declaration from Lagos port authority\n"
        f"- Proof of physical goods inspection\n\n"
        f"This is a mandatory compliance hold.\n\n"
        f"Regards,\nCompliance Department\n{t['company_lagos']}"
    )
    (email_dir / "urgent_sanctions_screening.eml").write_text(
        msg3.as_string(), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Email 4: David to Nadia (in response to compliance, privately) ---
    msg4 = EmailMessage()
    msg4["Subject"] = "Re: Routing adjustment needed"
    msg4["From"] = t["suspect_email"]
    msg4["To"] = t["broker_email"]
    msg4["Date"] = _thread_date(t, 19).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg4.set_content(
        f"Nadia,\n\n"
        f"Compliance is asking questions about the Dubai entity. We need to "
        f"route the second payment through your Odessa account instead.\n\n"
        f"Can you forward it onward from there? Use a different reference "
        f"for the second tranche.\n\n"
        f"Don't mention Rashid's company in any of the paperwork.\n\n"
        f"David"
    )
    (email_dir / "re_routing_adjustment.eml").write_text(
        msg4.as_string(), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Shipping invoice 1 ---
    inv_dir = base / "Documents" / "Financial" / "Invoices"
    inv_dir.mkdir(parents=True, exist_ok=True)
    invoice1 = (
        f"COMMERCIAL INVOICE\n\n"
        f"Invoice No: {t['ref_1']}\n"
        f"Date: {_thread_date(t, 10).strftime('%B %d, %Y')}\n\n"
        f"Shipper: {t['company_lagos']}\n"
        f"         14 Apapa Wharf Road, Lagos, Nigeria\n\n"
        f"Consignee: {t['company_dubai']}\n"
        f"           Jebel Ali Free Zone, Dubai, UAE\n\n"
        f"Via: {t['company_odessa']}\n"
        f"     Primorskiy Blvd 6, Odessa, Ukraine\n\n"
        f"{'Description':<35} {'Qty':>5} {'Unit Price':>12} {'Total':>12}\n"
        f"{'-' * 70}\n"
        f"{'Industrial pump assemblies':<35} {'12':>5} {'$18,333.33':>12} {'$220,000.00':>12}\n"
        f"{'Hydraulic control valves':<35} {'24':>5} {'$3,750.00':>12} {'$90,000.00':>12}\n"
        f"{'Freight & insurance':<35} {'':>5} {'':>12} {'$30,000.00':>12}\n"
        f"{'-' * 70}\n"
        f"{'TOTAL':>52} {'$340,000.00':>12}\n\n"
        f"Payment terms: Wire transfer prior to loading\n"
        f"Bank: {t['bank_dubai']}\n"
        f"IBAN: {t['iban_dubai']}\n"
        f"SWIFT: {t['swift_dubai']}\n"
        f"Reference: {t['ref_1']}\n"
    )
    (inv_dir / "commercial_invoice_pump_assemblies.txt").write_text(
        invoice1, encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Shipping invoice 2 ---
    invoice2 = (
        f"COMMERCIAL INVOICE\n\n"
        f"Invoice No: {t['ref_2']}\n"
        f"Date: {_thread_date(t, 45).strftime('%B %d, %Y')}\n\n"
        f"Shipper: {t['company_lagos']}\n"
        f"         14 Apapa Wharf Road, Lagos, Nigeria\n\n"
        f"Consignee: {t['company_odessa']}\n"
        f"           Primorskiy Blvd 6, Odessa, Ukraine\n\n"
        f"{'Description':<35} {'Qty':>5} {'Unit Price':>12} {'Total':>12}\n"
        f"{'-' * 70}\n"
        f"{'Generator sets (diesel)':<35} {'8':>5} {'$25,000.00':>12} {'$200,000.00':>12}\n"
        f"{'Electrical switchgear':<35} {'16':>5} {'$3,437.50':>12} {'$55,000.00':>12}\n"
        f"{'Freight & handling':<35} {'':>5} {'':>12} {'$20,000.00':>12}\n"
        f"{'-' * 70}\n"
        f"{'TOTAL':>52} {'$275,000.00':>12}\n\n"
        f"Payment terms: Wire transfer prior to loading\n"
        f"Bank: {t['bank_odessa']}\n"
        f"IBAN: {t['iban_odessa']}\n"
        f"Reference: {t['ref_2']}\n"
    )
    (inv_dir / "commercial_invoice_generator_sets.txt").write_text(
        invoice2, encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Transaction ledger CSV ---
    fin_dir = base / "Documents" / "Financial"
    fin_dir.mkdir(parents=True, exist_ok=True)
    path = fin_dir / "transactions_Q3_2024.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "From", "To", "Amount", "Currency", "Reference", "Type"])
        writer.writerow([
            _thread_date(t, 16).strftime("%Y-%m-%d"),
            t["company_lagos"], t["company_dubai"],
            t["amount_1"], "USD", t["ref_1"], "Wire Transfer",
        ])
        writer.writerow([
            _thread_date(t, 50).strftime("%Y-%m-%d"),
            t["company_lagos"], t["company_odessa"],
            t["amount_2"], "USD", t["ref_2"], "Wire Transfer",
        ])
        writer.writerow([
            _thread_date(t, 55).strftime("%Y-%m-%d"),
            t["company_odessa"], t["company_dubai"],
            t["amount_2"], "USD", t["ref_2"], "Wire Transfer",
        ])
    count += 1; progress.advance(task)

    # --- Bank statement (Emirates NBD) ---
    path = fin_dir / "statement_emirates_nbd_Q3.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Description", "Debit", "Credit", "Balance"])
        balance = 87_320.00
        writer.writerow([_thread_date(t, 5).strftime("%Y-%m-%d"),
                         "Opening balance", "", "", f"{balance:.2f}"])
        balance += t["amount_1"]
        writer.writerow([
            _thread_date(t, 16).strftime("%Y-%m-%d"),
            f"SWIFT Credit - {t['company_lagos']} - Ref {t['ref_1']}",
            "", f"{t['amount_1']:.2f}", f"{balance:.2f}",
        ])
        writer.writerow([
            _thread_date(t, 22).strftime("%Y-%m-%d"),
            "Transfer to operating account - Al-Mansoori Holdings",
            f"{300_000:.2f}", "", f"{balance - 300_000:.2f}",
        ])
        balance = balance - 300_000 + t["amount_2"]
        writer.writerow([
            _thread_date(t, 55).strftime("%Y-%m-%d"),
            f"SWIFT Credit - {t['company_odessa']} - Ref {t['ref_2']}",
            "", f"{t['amount_2']:.2f}", f"{balance:.2f}",
        ])
    count += 1; progress.advance(task)

    # --- Freight forwarding contract ---
    biz_dir = base / "Documents" / "Business"
    biz_dir.mkdir(parents=True, exist_ok=True)
    contract = (
        f"FREIGHT FORWARDING AGREEMENT\n\n"
        f"Date: {_thread_date(t, 3).strftime('%B %d, %Y')}\n\n"
        f"BETWEEN:\n"
        f"  {t['company_lagos']} (\"Shipper\")\n"
        f"  14 Apapa Wharf Road, Lagos, Nigeria\n\n"
        f"AND:\n"
        f"  {t['company_odessa']} (\"Forwarder\")\n"
        f"  Primorskiy Blvd 6, Odessa, Ukraine\n\n"
        f"1. SERVICES\n"
        f"   Forwarder shall arrange maritime transport of industrial equipment\n"
        f"   from Lagos port to final destination via Odessa transshipment hub.\n\n"
        f"2. COMPENSATION\n"
        f"   Shipper shall pay Forwarder a handling fee of 8% of declared cargo value\n"
        f"   for each shipment, plus port fees and insurance.\n\n"
        f"3. ROUTING\n"
        f"   All shipments shall transit through {t['company_odessa']} facilities\n"
        f"   in Odessa before final delivery to consignee.\n\n"
        f"4. PAYMENT\n"
        f"   All payments for end-consignee charges shall be directed to:\n"
        f"   {t['company_dubai']}\n"
        f"   {t['bank_dubai']}\n"
        f"   IBAN: {t['iban_dubai']}\n\n"
        f"5. CONFIDENTIALITY\n"
        f"   Both parties agree to maintain strict confidentiality regarding\n"
        f"   shipment details, routing, and financial arrangements.\n\n"
        f"SIGNED:\n"
        f"_________________________          _________________________\n"
        f"{t['suspect']}                     {t['broker']}\n"
        f"{t['company_lagos']}               {t['company_odessa']}\n"
    )
    (biz_dir / "freight_forwarding_agreement.txt").write_text(contract, encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Bill of lading (suspicious -- no customs stamps) ---
    bol = (
        f"BILL OF LADING\n\n"
        f"B/L No: BOL-{_thread_date(t, 20).strftime('%Y%m%d')}-001\n"
        f"Date: {_thread_date(t, 20).strftime('%B %d, %Y')}\n\n"
        f"Shipper: {t['company_lagos']}\n"
        f"Consignee: {t['company_dubai']}\n"
        f"Notify Party: {t['company_odessa']}\n\n"
        f"Vessel: MV Northern Promise\n"
        f"Port of Loading: Apapa Port, Lagos\n"
        f"Port of Discharge: Port Rashid, Dubai\n"
        f"Via: Odessa, Ukraine (transshipment)\n\n"
        f"{'Marks & Numbers':<25} {'Description':<30} {'Qty':>5} {'Weight (kg)':>12}\n"
        f"{'-' * 75}\n"
        f"{'TGFS-2024-001':<25} {'Industrial pump assemblies':<30} {'12':>5} {'4,800':>12}\n"
        f"{'TGFS-2024-002':<25} {'Hydraulic control valves':<30} {'24':>5} {'1,200':>12}\n"
        f"{'-' * 75}\n\n"
        f"Container: MSKU 773429-1 (40ft HC)\n"
        f"Seal No: TG-88421\n\n"
        f"Freight: PREPAID\n"
        f"Declared Value: ${t['amount_1']:,} USD\n\n"
        f"NOTE: Customs clearance documentation pending.\n"
        f"      Inspection certificate: NOT ATTACHED\n"
    )
    (biz_dir / "bill_of_lading_northern_promise.txt").write_text(bol, encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Chat log: David and Nadia coordination ---
    chat_dir = base / "Communications" / "ChatLogs"
    chat_dir.mkdir(parents=True, exist_ok=True)
    chat_dn = [
        {"timestamp": _thread_date(t, 8).isoformat(), "sender": "David",
         "phone": "+234-802-555-0194",
         "message": "Nadia, is the Odessa warehouse ready? Need the transit docs for the first shipment."},
        {"timestamp": _thread_date(t, 8).replace(hour=14).isoformat(), "sender": "Nadia",
         "phone": "+380-50-555-0712",
         "message": "Ready. But there are no actual goods to receive, correct? Just the paperwork?"},
        {"timestamp": _thread_date(t, 9).isoformat(), "sender": "David",
         "phone": "+234-802-555-0194",
         "message": "Correct. Container will be sealed in Lagos. Don't open it in Odessa. Just stamp the transit docs and forward to Dubai."},
        {"timestamp": _thread_date(t, 9).replace(hour=16).isoformat(), "sender": "Nadia",
         "phone": "+380-50-555-0712",
         "message": "Understood. Rashid says the Dubai account is ready for the wire."},
        {"timestamp": _thread_date(t, 15).isoformat(), "sender": "David",
         "phone": "+234-802-555-0194",
         "message": "Wire sent. $340K to Emirates NBD. Reference SHP-2281-PF. Confirm when it lands."},
        {"timestamp": _thread_date(t, 17).isoformat(), "sender": "Nadia",
         "phone": "+380-50-555-0712",
         "message": "Rashid confirmed receipt. Second shipment in 30 days."},
        {"timestamp": _thread_date(t, 18).isoformat(), "sender": "David",
         "phone": "+234-802-555-0194",
         "message": "Compliance is sniffing around the Dubai entity. Route the second payment through your account in Odessa."},
        {"timestamp": _thread_date(t, 18).replace(hour=20).isoformat(), "sender": "Nadia",
         "phone": "+380-50-555-0712",
         "message": "I can do that. My ПриватБанк account. I'll forward it to Rashid from there."},
    ]
    (chat_dir / "chat_export_012.json").write_text(
        json.dumps(chat_dn, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Chat log: Nadia and Rashid (Russian/Arabic mix) ---
    chat_nr = [
        {"timestamp": _thread_date(t, 10).isoformat(), "sender": "Надя",
         "phone": "+380-50-555-0712",
         "message": "Рашид, документы на отправку готовы. Контейнер пустой, только бумаги.", "language": "ru"},
        {"timestamp": _thread_date(t, 10).replace(hour=15).isoformat(), "sender": "راشد",
         "phone": "+971-55-555-0839",
         "message": "مفهوم. حساب الإمارات دبي الوطني جاهز. الحوالة متى؟", "language": "ar"},
        {"timestamp": _thread_date(t, 11).isoformat(), "sender": "Надя",
         "phone": "+380-50-555-0712",
         "message": "Давид говорит на следующей неделе. $340,000 за первую партию.", "language": "ru"},
        {"timestamp": _thread_date(t, 16).replace(hour=19).isoformat(), "sender": "راشد",
         "phone": "+971-55-555-0839",
         "message": f"تم استلام المبلغ. المرجع {t['ref_1']}. سأحول $300K إلى حساب المنصوري القابضة.", "language": "ar"},
        {"timestamp": _thread_date(t, 50).isoformat(), "sender": "Надя",
         "phone": "+380-50-555-0712",
         "message": f"Вторая партия через мой счёт. {t['ref_2']}. $275,000. Переведу тебе через ПриватБанк.", "language": "ru"},
        {"timestamp": _thread_date(t, 56).isoformat(), "sender": "راشد",
         "phone": "+971-55-555-0839",
         "message": "تم. احذف هذه المحادثة.", "language": "ar"},
    ]
    (chat_dir / "chat_export_013.json").write_text(
        json.dumps(chat_nr, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Notes: routing memo ---
    notes_dir = base / "Documents" / "Notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    memo = (
        f"Shipping arrangements:\n\n"
        f"Shipment 1:\n"
        f"  {t['ref_1']} - $340K\n"
        f"  Lagos -> Odessa -> Dubai\n"
        f"  Payment direct to Emirates NBD ****{t['account_tail']}\n"
        f"  IBAN: {t['iban_dubai']}\n\n"
        f"Shipment 2:\n"
        f"  {t['ref_2']} - $275K\n"
        f"  Lagos -> Odessa (Nadia's account) -> Dubai\n"
        f"  Route through ПриватБанк to avoid compliance flag\n\n"
        f"Total: $615,000\n"
        f"No customs declarations filed.\n"
        f"Container sealed in Lagos -- do NOT open in transit.\n"
        f"Destroy after completion.\n"
    )
    (notes_dir / "shipping_arrangements.txt").write_text(memo, encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Warehouse photo (empty -- the goods don't exist) ---
    media_dir = base / "Media" / "Photos"
    media_dir.mkdir(parents=True, exist_ok=True)
    warehouse_text = (
        f"WAREHOUSE INSPECTION PHOTO\n\n"
        f"Location: Apapa Port Warehouse B7\n"
        f"Date: {_thread_date(t, 19).strftime('%B %d, %Y')}\n\n"
        f"Container: MSKU 773429-1\n"
        f"Status: SEALED\n\n"
        f"[Empty warehouse bay]\n"
        f"[No equipment visible]\n"
        f"[Loading dock vacant]\n"
    )
    img_bytes = _make_image(warehouse_text, 800, 600, bg="#d4c9a8")
    (media_dir / "warehouse_inspection_B7.jpg").write_bytes(img_bytes)
    count += 1; progress.advance(task)

    # --- Shipping portal screenshot ---
    screenshots_dir = base / "Media" / "Screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    portal_text = (
        f"TRANSGLOBAL FREIGHT - SHIPMENT TRACKER\n\n"
        f"Shipment: {t['ref_1']}\n"
        f"Status: IN TRANSIT\n\n"
        f"Origin: Lagos, NG\n"
        f"Destination: Dubai, AE\n"
        f"Via: Odessa, UA\n\n"
        f"Container: MSKU 773429-1\n"
        f"Vessel: MV Northern Promise\n\n"
        f"Timeline:\n"
        f"  {_thread_date(t, 20).strftime('%d %b')} - Departed Lagos\n"
        f"  {_thread_date(t, 35).strftime('%d %b')} - Arrived Odessa\n"
        f"  {_thread_date(t, 38).strftime('%d %b')} - Departed Odessa\n"
        f"  {_thread_date(t, 52).strftime('%d %b')} - ETA Dubai\n\n"
        f"Customs Status: PENDING\n"
        f"Inspection: NOT SCHEDULED\n"
    )
    img = Image.new("RGB", (800, 600), "#1a2634")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except OSError:
            font = ImageFont.load_default()
    y = 20
    for line in portal_text.split("\n"):
        draw.text((20, y), line, fill="#00cc88", font=font)
        y += 22
    buf = BytesIO()
    img.save(buf, format="PNG")
    (screenshots_dir / "shipment_tracker_screenshot.png").write_bytes(buf.getvalue())
    count += 1; progress.advance(task)

    # --- Browser history: sanctions screening visits ---
    digital_dir = base / "AppData"
    digital_dir.mkdir(parents=True, exist_ok=True)
    history = [
        {"url": "https://sanctionssearch.ofac.treas.gov/", "title": "OFAC Sanctions List Search",
         "visited_at": _thread_date(t, 6).isoformat()},
        {"url": "https://sanctionssearch.ofac.treas.gov/", "title": "OFAC Sanctions List Search",
         "visited_at": _thread_date(t, 17).isoformat()},
        {"url": "https://www.emiratesnbd.com/en/business-banking/trade-finance",
         "title": "Trade Finance - Emirates NBD",
         "visited_at": _thread_date(t, 13).isoformat()},
        {"url": "https://www.marinetraffic.com/en/ais/details/ships/mv-northern-promise",
         "title": "MV Northern Promise - MarineTraffic",
         "visited_at": _thread_date(t, 21).isoformat()},
        {"url": "https://www.marinetraffic.com/en/ais/details/ships/mv-northern-promise",
         "title": "MV Northern Promise - MarineTraffic",
         "visited_at": _thread_date(t, 36).isoformat()},
        {"url": "https://comtrade.un.org/data/", "title": "UN Comtrade Database",
         "visited_at": _thread_date(t, 7).isoformat()},
    ]
    (digital_dir / "browser_history_work.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    count += 1; progress.advance(task)

    # --- Second bank statement (PrivatBank showing forwarding) ---
    path = fin_dir / "statement_privatbank_Q4.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Description", "Debit", "Credit", "Balance"])
        balance = 42_150.00
        writer.writerow([_thread_date(t, 40).strftime("%Y-%m-%d"),
                         "Opening balance", "", "", f"{balance:.2f}"])
        balance += t["amount_2"]
        writer.writerow([
            _thread_date(t, 50).strftime("%Y-%m-%d"),
            f"SWIFT Credit - {t['company_lagos']} - Ref {t['ref_2']}",
            "", f"{t['amount_2']:.2f}", f"{balance:.2f}",
        ])
        balance -= t["amount_2"]
        writer.writerow([
            _thread_date(t, 52).strftime("%Y-%m-%d"),
            f"SWIFT Transfer - {t['company_dubai']} - Freight forwarding",
            f"{t['amount_2']:.2f}", "", f"{balance:.2f}",
        ])
    count += 1; progress.advance(task)

    # --- Email 5: Nadia confirming second payment forwarded (Russian) ---
    msg5 = EmailMessage()
    msg5["Subject"] = "Переказ виконано"
    msg5["From"] = t["broker_email"]
    msg5["To"] = t["suspect_email"]
    msg5["Date"] = _thread_date(t, 53).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg5["X-Language"] = "uk"
    msg5.set_content(
        f"David,\n\n"
        f"Переказ $275,000 на рахунок GulfStar в Дубаї виконано.\n"
        f"Референс: {t['ref_2']}\n"
        f"Через мій рахунок в ПриватБанку, як домовлялись.\n\n"
        f"Рашид підтвердить отримання.\n\n"
        f"Надія"
    )
    (email_dir / "perekaz_vykonano.eml").write_text(msg5.as_string(), encoding="utf-8")
    count += 1; progress.advance(task)

    # --- Sanctions screening result (saved HTML as text) ---
    downloads_dir = base / "Downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    screening = (
        f"OFAC SDN Search Results\n"
        f"Query: \"GulfStar Trading\"\n"
        f"Date: {_thread_date(t, 17).strftime('%Y-%m-%d')}\n\n"
        f"Results: 1 potential match\n\n"
        f"  Name: GULFSTAR TRADING FZE\n"
        f"  Type: Entity\n"
        f"  Program: SDGT\n"
        f"  Remarks: Linked to designated individual.\n"
        f"  Address: Jebel Ali Free Zone, Dubai, UAE\n\n"
        f"--- END OF RESULTS ---\n"
    )
    (downloads_dir / "ofac_search_gulfstar.txt").write_text(screening, encoding="utf-8")
    count += 1; progress.advance(task)

    return count


def generate_workshop_evidence(base: Path, progress, task) -> int:
    """Generate the workshop scenario evidence thread."""
    return _generate_phantom_freight(base, progress, task)


# ---------------------------------------------------------------------------
# Noise generation: realistic irrelevant content to bury investigative signal
# ---------------------------------------------------------------------------
# A real seized hard drive is ~75% mundane personal/work content.  These
# functions generate ~500 noise files across 5 categories: downloaded media,
# personal life content, mundane work docs, digital clutter, and red herrings.

_PICSUM_TIMEOUT = 8.0
_WIKI_TIMEOUT = 6.0

_WIKIPEDIA_TOPICS = [
    "Pasta", "Association_football", "Solar_System", "Python_(programming_language)",
    "History_of_chess", "Blue_whale", "Great_Wall_of_China", "Jazz", "Mount_Everest",
    "Leonardo_da_Vinci", "Coffee", "Origami", "Tour_de_France", "Coral_reef",
    "History_of_photography", "Bonsai", "Sushi", "Aurora_borealis", "Table_tennis",
    "Machu_Picchu", "Yoga", "National_parks_of_the_United_States", "Fermentation",
    "History_of_the_bicycle", "Honey_bee", "Gardening", "Board_game", "Tsunami",
    "Architecture_of_ancient_Rome", "Chocolate",
]

_PERSONAL_EMAIL_SUBJECTS = [
    "Re: Dinner tonight?", "Kid's soccer schedule this week",
    "Fwd: Netflix recommendation - you HAVE to watch this",
    "Dentist appointment reminder - Thursday 2pm",
    "Happy Birthday!!! 🎂", "Weekend BBQ plans",
    "Fwd: Flight confirmation - JFK to LAX",
    "Dog walker schedule update", "Running group Saturday 7am",
    "Re: Book club pick for next month", "Grocery list from Mom",
    "Your Amazon order has shipped", "Gym membership renewal",
    "Fwd: School newsletter - March 2024", "Tickets for the concert!",
    "Plumber coming Tuesday morning", "Recipe: Grandma's lasagna",
    "Carpool schedule change", "Re: Holiday photos",
    "Your Spotify Wrapped is here!", "Oil change reminder - Honda",
    "HOA meeting minutes", "Re: Vacation rental inquiry",
    "Fwd: 50% off at Target this weekend", "PTA meeting next Tuesday",
    "Re: Fantasy football draft picks", "Parking permit renewal notice",
    "Fwd: Funny video - cats vs cucumbers", "Re: Thanksgiving plans",
    "Lunch tomorrow? Thai place?",
]

_PERSONAL_EMAIL_BODIES = [
    "Hey!\n\nAre we still on for dinner tonight? I was thinking that new Italian place on Main St. "
    "Let me know if 7pm works.\n\nSee you there!\n{name}",
    "Hi everyone,\n\nJust a reminder that practice is at 4:30pm on Wednesday and the game is "
    "Saturday at 10am. Please bring water bottles and shin guards.\n\nCoach {name}",
    "You seriously need to watch this show. Season 2 just dropped and it's even better than "
    "the first. Trust me on this one.\n\n{name}",
    "This is a reminder of your upcoming appointment:\n\nDr. {name}\n{date}\n\n"
    "Please arrive 15 minutes early to complete paperwork.",
    "HAPPY BIRTHDAY! Hope you have an amazing day! We should celebrate this weekend - "
    "drinks on me! 🎉\n\nLove,\n{name}",
    "Hey,\n\nWeather looks great for Saturday. I'll fire up the grill around noon. Bring "
    "whatever you want to drink. I've got burgers and hot dogs covered.\n\n{name}",
    "Forwarding this for your records. Flight departs 6:45am, arrives 9:30am local time. "
    "Seat 14C. Confirmation: {ref}\n\n{name}",
    "Reminder: {name} from Waggy Tails will pick up the dog at 11am on weekdays. "
    "Key is under the mat. Please leave water bowl filled.",
    "Quick reminder - we're meeting at the trail head at 7am sharp this Saturday. "
    "5K easy pace then coffee after. Don't be late!\n\n{name}",
    "So I'm voting for the new {name} novel for book club. Anyone else read it yet? "
    "Fair warning - it's 400+ pages.\n\nLet me know!",
]

_WORK_EMAIL_SUBJECTS = [
    "PTO Request - {date}", "All-Hands Meeting - Q{q} Review",
    "IT: Password Reset Required by Friday",
    "Re: Standup notes - {date}", "Updated: Conference Room B reserved",
    "Fwd: Company picnic - RSVP needed",
    "Reminder: Expense reports due end of month",
    "New employee onboarding - Welcome {name}!",
    "Re: Quarterly OKR check-in", "IT: VPN maintenance window tonight",
    "Fwd: Benefits enrollment deadline approaching",
    "Team lunch - Friday noon", "Re: Printer on 3rd floor is jammed again",
    "HR: Updated PTO policy effective Jan 1",
    "Parking lot B closed for repaving next week",
    "Re: Draft slides for client presentation",
    "Fire drill scheduled for Thursday 2pm",
    "Fwd: Industry conference - early bird registration",
    "Re: Can someone cover my shift Friday?",
    "Monthly team retrospective - action items",
]

_WORK_EMAIL_BODIES = [
    "Hi Manager,\n\nI'd like to request PTO from {date} to {date2}. "
    "All my projects are on track and {name} has agreed to cover.\n\nThanks,\n{sender}",
    "Team,\n\nReminder that the All-Hands is tomorrow at 10am in the main conference room. "
    "We'll be reviewing Q{q} results and sharing the roadmap for next quarter.\n\n{sender}",
    "Your network password expires in 3 days. Please update it through the IT portal "
    "at https://it.internal/password-reset. Requirements: 12+ chars, one symbol, one number.\n\n"
    "IT Help Desk",
    "Standup notes for {date}:\n- {name}: Finishing the API migration, blocked on DB access\n"
    "- {sender}: Code review for PR #442, starting QA testing\n- {name2}: Out sick today",
    "All,\n\nJust a reminder to submit your expense reports by the end of the month. "
    "Please attach all receipts. Reports submitted late will be processed in the next cycle.\n\n"
    "Finance Team",
]

_SOCIAL_PLATFORMS = ["Instagram", "Twitter", "Facebook", "TikTok", "LinkedIn"]

_RECIPE_TITLES = {
    "en": ["Classic Banana Bread", "Chicken Tikka Masala", "Caesar Salad",
            "Homemade Pizza Dough", "Chocolate Chip Cookies"],
    "es": ["Paella Valenciana", "Guacamole Casero", "Tacos al Pastor",
            "Arroz con Leche", "Enchiladas Verdes"],
    "de": ["Wiener Schnitzel", "Kartoffelsalat", "Schwarzwälder Kirschtorte",
            "Bratkartoffeln", "Apfelstrudel"],
    "ja": ["味噌ラーメン", "唐揚げ", "親子丼", "抹茶ティラミス", "焼きそば"],
    "pt": ["Brigadeiro", "Feijoada", "Pão de Queijo", "Coxinha", "Açaí Bowl"],
}

_NEWS_HEADLINES = [
    "City Council Approves New Park Renovation Project",
    "Local High School Wins State Championship in Basketball",
    "Weather Advisory: Heavy Rain Expected This Weekend",
    "Tech Company Opens New Office Downtown, Creating 200 Jobs",
    "Annual Food Festival Returns to Waterfront Next Month",
    "Highway Construction to Cause Delays Through Summer",
    "Public Library Launches Free Coding Workshops for Kids",
    "Community Garden Project Seeks Volunteers for Spring Planting",
    "New Restaurant Review: Mediterranean Cuisine on Elm Street",
    "Local Artist Exhibition Opens at City Gallery",
]


def _download_image(url: str, timeout: float = _PICSUM_TIMEOUT) -> bytes | None:
    """Download an image from a URL, returning bytes or None on failure."""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url)
            if resp.status_code == 200 and len(resp.content) > 100:
                return resp.content
    except Exception:
        pass
    return None


def _download_wikipedia_summary(topic: str) -> str | None:
    """Fetch a Wikipedia article summary via the public REST API."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
    headers = {"User-Agent": "ForensicTriageTool/0.1 (educational demo; no scraping)"}
    try:
        with httpx.Client(timeout=_WIKI_TIMEOUT, headers=headers) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title", topic)
                extract = data.get("extract", "")
                if extract:
                    return f"{title}\n{'=' * len(title)}\n\n{extract}"
    except Exception:
        pass
    return None


def _noise_placeholder_image(label: str, w: int = 640, h: int = 480) -> bytes:
    """Generate a colored placeholder image when downloads fail."""
    bg_colors = ["#4a90d9", "#d9534f", "#5cb85c", "#f0ad4e", "#5bc0de",
                 "#8e44ad", "#2c3e50", "#e74c3c", "#27ae60", "#f39c12"]
    bg = random.choice(bg_colors)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, h // 2 - 10), label, fill="white", font=font)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _noise_downloaded_media(base: Path, progress, task) -> int:
    """Download real images from Lorem Picsum and Wikipedia articles."""
    count = 0

    photo_dirs = {
        "personal": base / "Personal" / "Photos",
        "vacation": base / "Media" / "Photos" / "vacation",
        "misc": base / "Media" / "Photos" / "misc",
    }
    for d in photo_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    personal_labels = [
        "family_bbq", "dog_park", "beach_sunset", "birthday_party",
        "hiking_trip", "garden", "cat_sleeping", "road_trip",
        "snow_day", "backyard_pool", "friends_dinner", "morning_coffee",
        "farmers_market", "bike_ride", "game_night",
    ]
    vacation_labels = [
        "hotel_pool", "beach_panorama", "restaurant_view", "sunset_cruise",
        "mountain_vista", "city_skyline", "temple_visit", "street_market",
        "waterfall", "snorkeling", "train_window", "resort_lobby",
        "local_food", "souvenir_shop", "airport_lounge",
    ]
    misc_labels = [
        "random_screenshot", "meme_funny", "product_photo", "flyer_scan",
        "parking_ticket", "business_card", "whiteboard_notes", "package_label",
        "store_receipt_photo", "menu_photo",
    ]

    resolutions = [(640, 480), (800, 600), (1024, 768), (1280, 960), (480, 640)]

    all_downloads = []
    for label in personal_labels:
        all_downloads.append(("personal", label))
    for label in vacation_labels:
        all_downloads.append(("vacation", label))
    for label in misc_labels:
        all_downloads.append(("misc", label))

    for category, label in all_downloads:
        w, h = random.choice(resolutions)
        img_bytes = _download_image(f"https://picsum.photos/{w}/{h}")
        if img_bytes is None:
            img_bytes = _noise_placeholder_image(label.replace("_", " ").title(), w, h)
        suffix = random.choice([".jpg", ".jpg", ".jpg", ".png"])
        date_str = _random_date().strftime("%Y%m%d_%H%M%S")
        fname = f"IMG_{date_str}_{label}{suffix}"
        out_dir = photo_dirs[category]
        out_path = out_dir / fname
        out_path.write_bytes(img_bytes)
        count += 1
        progress.advance(task)

    ref_dir = base / "Documents" / "Reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = base / "Downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    for topic in random.sample(_WIKIPEDIA_TOPICS, min(20, len(_WIKIPEDIA_TOPICS))):
        text = _download_wikipedia_summary(topic)
        if text is None:
            lf = LOCALE_FAKERS["en"]
            text = f"{topic.replace('_', ' ')}\n{'=' * 30}\n\n{lf.paragraph(nb_sentences=10)}"
        dest = ref_dir if random.random() > 0.4 else dl_dir
        fname = f"{topic.lower().replace('_', '-')}.txt"
        (dest / fname).write_text(text, encoding="utf-8")
        count += 1
        progress.advance(task)

    for i in range(30):
        lf = LOCALE_FAKERS[_pick_locale()]
        title = lf.catch_phrase()
        body = "\n\n".join(lf.paragraph(nb_sentences=random.randint(3, 8)) for _ in range(random.randint(3, 7)))
        content = f"{title}\n\n{body}\n"
        (dl_dir / f"document_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    return count


def _noise_personal(base: Path, progress, task) -> int:
    """Generate mundane personal content: emails, photos, lists, recipes, etc."""
    count = 0

    # --- Personal emails (~30) ---
    email_dir = base / "Communications" / "Email"
    email_dir.mkdir(parents=True, exist_ok=True)
    for i in range(30):
        msg = EmailMessage()
        msg["Subject"] = random.choice(_PERSONAL_EMAIL_SUBJECTS)
        sender_name = fake_en.name()
        msg["From"] = f"{sender_name.lower().replace(' ', '.')}@{fake_en.free_email_domain()}"
        msg["To"] = f"{fake_en.user_name()}@{fake_en.free_email_domain()}"
        msg["Date"] = _random_date().strftime("%a, %d %b %Y %H:%M:%S +0000")
        body_tmpl = random.choice(_PERSONAL_EMAIL_BODIES)
        body = body_tmpl.format(
            name=fake_en.name(),
            date=_random_date().strftime("%B %d"),
            ref=fake_en.bothify("???###").upper(),
        )
        msg.set_content(body)
        (email_dir / f"personal_{i:03d}.eml").write_text(msg.as_string(), encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Personal photos (Pillow generated) (~20) ---
    personal_photo_dir = base / "Personal" / "Photos"
    personal_photo_dir.mkdir(parents=True, exist_ok=True)
    personal_photo_labels = [
        "Family BBQ July 4th", "Dog at the park", "Sunset beach trip",
        "Office holiday party", "First day of school", "Birthday cake",
        "Thanksgiving dinner", "Snow day!", "Backyard garden",
        "New puppy!!", "Soccer game Saturday", "Camping trip",
        "Friends reunion", "Graduation day", "Baby shower",
        "Kitchen remodel progress", "New car!", "Movie night",
        "Morning jog", "Weekend farmers market",
        "Picnic in the park", "Cat on the couch", "Fireworks show",
        "Beach volleyball", "Garden harvest",
    ]
    for i, label in enumerate(personal_photo_labels):
        img_bytes = _make_image(
            f"[Personal Photo]\n\n{label}\n\n{_random_date().strftime('%B %d, %Y')}",
            random.choice([640, 800, 1024]),
            random.choice([480, 600, 768]),
            bg=random.choice(["#e8f0e8", "#f0e8e8", "#e8e8f0", "#f0f0e0"]),
        )
        (personal_photo_dir / f"photo_{i:03d}.jpg").write_bytes(img_bytes)
        count += 1
        progress.advance(task)

    # --- Social media screenshots (~20) ---
    screenshots_dir = base / "Media" / "Screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        platform = random.choice(_SOCIAL_PLATFORMS)
        username = fake_en.user_name()
        post = fake_en.sentence(nb_words=random.randint(5, 15))
        likes = random.randint(3, 4500)
        comments = random.randint(0, 200)
        time_ago = f"{random.randint(1, 23)}h"
        text = (
            f"{'─' * 40}\n"
            f"  {platform}\n"
            f"{'─' * 40}\n\n"
            f"  @{username}  ·  {time_ago}\n\n"
            f"  {post}\n\n"
            f"  ♥ {likes:,}    💬 {comments}    ↗ Share\n"
            f"{'─' * 40}"
        )
        img = Image.new("RGB", (400, 500), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except OSError:
            font = ImageFont.load_default()
        y = 15
        for line in text.split("\n"):
            draw.text((10, y), line, fill="#333333", font=font)
            y += 20
        buf = BytesIO()
        img.save(buf, format="PNG")
        (screenshots_dir / f"social_{platform.lower()}_{i:03d}.png").write_bytes(buf.getvalue())
        count += 1
        progress.advance(task)

    # --- Grocery / shopping lists (~10) ---
    lists_dir = base / "Personal" / "Lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    grocery_items = [
        "Milk", "Eggs", "Bread", "Butter", "Chicken breast", "Rice",
        "Bananas", "Apples", "Tomatoes", "Onions", "Garlic", "Olive oil",
        "Pasta", "Cheddar cheese", "Yogurt", "Orange juice", "Coffee",
        "Paper towels", "Dish soap", "Laundry detergent", "Trash bags",
        "Dog food", "Cat litter", "Cereal", "Frozen pizza", "Ice cream",
        "Avocados", "Spinach", "Salmon", "Ground beef", "Tortillas",
    ]
    for i in range(10):
        items = random.sample(grocery_items, random.randint(6, 15))
        header = random.choice(["Grocery List", "Shopping List", "Need to buy", "Store run"])
        content = f"{header} - {_random_date().strftime('%m/%d')}\n{'─' * 20}\n"
        for item in items:
            qty = random.choice(["", "2x ", "1 lb ", "6 pack ", ""])
            content += f"□ {qty}{item}\n"
        (lists_dir / f"list_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Calendar/reminders (~5) ---
    calendar_dir = base / "Personal"
    calendar_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        events = []
        for _ in range(random.randint(8, 20)):
            dt = _random_date()
            events.append({
                "title": random.choice([
                    "Dentist appointment", "Gym - leg day", "Date night",
                    f"Dinner with {fake_en.first_name()}", "Car service",
                    "Parent-teacher conference", "Yoga class",
                    "Pick up dry cleaning", f"Call {fake_en.first_name()}",
                    "Vet appointment", "Hair appointment", "Oil change",
                ]),
                "start": dt.isoformat(),
                "end": (dt + timedelta(hours=1)).isoformat(),
                "location": random.choice(["", fake_en.address().replace("\n", ", "), ""]),
            })
        (calendar_dir / f"calendar_export_{i:03d}.json").write_text(
            json.dumps(events, indent=2), encoding="utf-8"
        )
        count += 1
        progress.advance(task)

    # --- Recipes (~15) ---
    recipe_dir = base / "Personal" / "Recipes"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        locale = random.choice(list(_RECIPE_TITLES.keys()))
        title = random.choice(_RECIPE_TITLES[locale])
        lf = LOCALE_FAKERS[locale]
        ingredients = "\n".join(f"- {lf.sentence(nb_words=3)}" for _ in range(random.randint(5, 12)))
        steps = "\n".join(f"{j+1}. {lf.sentence(nb_words=random.randint(6, 12))}"
                          for j in range(random.randint(4, 8)))
        content = f"{title}\n{'=' * len(title)}\n\nServings: {random.randint(2, 8)}\n"
        content += f"Prep time: {random.randint(10, 45)} min\nCook time: {random.randint(15, 90)} min\n\n"
        content += f"Ingredients:\n{ingredients}\n\nInstructions:\n{steps}\n"
        (recipe_dir / f"recipe_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Music playlists (~5) ---
    for i in range(5):
        tracks = []
        for _ in range(random.randint(15, 40)):
            tracks.append({
                "title": fake_en.catch_phrase(),
                "artist": fake_en.name(),
                "album": fake_en.catch_phrase(),
                "duration_ms": random.randint(120000, 360000),
            })
        playlist = {
            "name": random.choice(["Road Trip Mix", "Workout Jams", "Chill Vibes",
                                   "90s Nostalgia", "Study Music"]),
            "tracks": tracks,
            "created": _random_date().isoformat(),
        }
        (calendar_dir / f"playlist_{i:03d}.json").write_text(
            json.dumps(playlist, indent=2), encoding="utf-8"
        )
        count += 1
        progress.advance(task)

    # --- Contact vCards (~5) ---
    for i in range(5):
        name = fake_en.name()
        parts = name.split()
        fn, ln = parts[0], parts[-1]
        vcard = (
            f"BEGIN:VCARD\nVERSION:3.0\n"
            f"FN:{name}\nN:{ln};{fn};;;\n"
            f"TEL;TYPE=CELL:{fake_en.phone_number()}\n"
            f"EMAIL:{fake_en.email()}\n"
            f"ORG:{fake_en.company()}\n"
            f"END:VCARD\n"
        )
        (calendar_dir / f"contact_{i:03d}.vcf").write_text(vcard, encoding="utf-8")
        count += 1
        progress.advance(task)

    return count


def _noise_work(base: Path, progress, task) -> int:
    """Generate mundane work documents: HR emails, agendas, timesheets, logs."""
    count = 0

    # --- Work emails (~20) ---
    email_dir = base / "Communications" / "Email"
    email_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        msg = EmailMessage()
        subj_tmpl = random.choice(_WORK_EMAIL_SUBJECTS)
        sender_name = fake_en.name()
        name2 = fake_en.name()
        subj = subj_tmpl.format(
            date=_random_date().strftime("%B %d"),
            q=random.randint(1, 4),
            name=fake_en.first_name(),
        )
        msg["Subject"] = subj
        domain = random.choice(["acme-corp.com", "globaltech.io", "initech.com", "vandelay-ind.com"])
        msg["From"] = f"{sender_name.lower().replace(' ', '.')}@{domain}"
        msg["To"] = f"{fake_en.user_name()}@{domain}"
        msg["Date"] = _random_date().strftime("%a, %d %b %Y %H:%M:%S +0000")
        body_tmpl = random.choice(_WORK_EMAIL_BODIES)
        body = body_tmpl.format(
            name=fake_en.first_name(),
            name2=name2,
            date=_random_date().strftime("%B %d"),
            date2=(_random_date() + timedelta(days=random.randint(1, 5))).strftime("%B %d"),
            q=random.randint(1, 4),
            sender=sender_name.split()[0],
        )
        msg.set_content(body)
        (email_dir / f"work_{i:03d}.eml").write_text(msg.as_string(), encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Meeting agendas (~10) ---
    work_dir = base / "Work" / "Projects"
    work_dir.mkdir(parents=True, exist_ok=True)
    agenda_topics = [
        "Q{q} OKR Review", "Weekly Standup", "Sprint Retrospective",
        "Product Roadmap Planning", "Budget Review Meeting",
        "Design Review - Mobile App", "Security Audit Follow-up",
        "New Hire Orientation", "Customer Success Sync", "Engineering All-Hands",
    ]
    for i in range(15):
        topic = random.choice(agenda_topics).format(q=random.randint(1, 4))
        attendees = [fake_en.name() for _ in range(random.randint(3, 8))]
        items = [fake_en.sentence(nb_words=random.randint(4, 8)) for _ in range(random.randint(3, 6))]
        content = f"MEETING AGENDA: {topic}\n"
        content += f"Date: {_random_date().strftime('%B %d, %Y')}\n"
        content += f"Time: {random.choice(['9:00 AM', '10:00 AM', '2:00 PM', '3:30 PM'])}\n"
        content += f"Attendees: {', '.join(attendees)}\n\n"
        for j, item in enumerate(items, 1):
            content += f"{j}. {item}\n"
        content += f"\nAction items from last meeting:\n"
        for _ in range(random.randint(1, 3)):
            content += f"  - {fake_en.name()}: {fake_en.sentence(nb_words=6)}\n"
        (work_dir / f"agenda_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Resumes (~5) ---
    hr_dir = base / "Work" / "HR"
    hr_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        name = fake_en.name()
        content = f"{name}\n{'=' * len(name)}\n"
        content += f"{fake_en.email()}  |  {fake_en.phone_number()}  |  {fake_en.city()}, {fake_en.state_abbr()}\n\n"
        content += f"SUMMARY\n{fake_en.paragraph(nb_sentences=3)}\n\n"
        content += "EXPERIENCE\n"
        for _ in range(random.randint(2, 4)):
            content += f"\n{fake_en.job()} - {fake_en.company()}\n"
            content += f"{_random_date().strftime('%B %Y')} - Present\n"
            content += f"  - {fake_en.sentence(nb_words=8)}\n"
            content += f"  - {fake_en.sentence(nb_words=7)}\n"
        content += "\nEDUCATION\n"
        content += f"B.S. in {random.choice(['Computer Science', 'Business Administration', 'Marketing', 'Engineering'])}\n"
        content += f"{fake_en.company()} University, {random.randint(2005, 2020)}\n"
        (hr_dir / f"resume_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Presentation notes (~10) ---
    pres_dir = base / "Work" / "Presentations"
    pres_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        title = fake_en.catch_phrase()
        slides = []
        for s in range(random.randint(5, 12)):
            slides.append(f"Slide {s+1}: {fake_en.sentence(nb_words=random.randint(3, 6))}\n"
                          f"  - {fake_en.sentence(nb_words=6)}\n"
                          f"  - {fake_en.sentence(nb_words=5)}")
        content = f"Presentation: {title}\nDate: {_random_date().strftime('%B %d, %Y')}\n\n"
        content += "\n\n".join(slides)
        (pres_dir / f"presentation_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Expense reports & timesheets (~15 CSV) ---
    for i in range(15):
        path = work_dir / f"{'expense_report' if i < 8 else 'timesheet'}_{i:03d}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if i < 8:
                writer.writerow(["Date", "Category", "Description", "Amount", "Receipt"])
                for _ in range(random.randint(5, 20)):
                    writer.writerow([
                        _random_date().strftime("%Y-%m-%d"),
                        random.choice(["Meals", "Travel", "Office Supplies", "Software", "Parking"]),
                        fake_en.sentence(nb_words=4),
                        round(random.uniform(5, 250), 2),
                        random.choice(["Yes", "Yes", "No", "Pending"]),
                    ])
            else:
                writer.writerow(["Date", "Project", "Hours", "Description"])
                for _ in range(random.randint(10, 25)):
                    writer.writerow([
                        _random_date().strftime("%Y-%m-%d"),
                        random.choice(["Project Alpha", "Client Portal", "Internal Tools", "Bug Fixes", "Meetings"]),
                        round(random.uniform(0.5, 8), 1),
                        fake_en.sentence(nb_words=5),
                    ])
        count += 1
        progress.advance(task)

    # --- IT / system logs (~15) ---
    logs_dir = base / "AppData" / "Logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        lines = []
        base_dt = _random_date()
        for j in range(random.randint(50, 200)):
            ts = (base_dt + timedelta(seconds=j * random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S")
            level = random.choice(["INFO", "INFO", "INFO", "WARN", "DEBUG", "ERROR"])
            source = random.choice(["kernel", "sshd", "systemd", "cron", "apache2",
                                    "NetworkManager", "WindowsUpdate", "Bluetooth"])
            msg_text = random.choice([
                "Connection established from 192.168.1." + str(random.randint(1, 254)),
                "Service started successfully",
                "Disk usage at " + str(random.randint(30, 85)) + "%",
                "Package updated: " + fake_en.bothify("lib????-#.#.#"),
                "User login: " + fake_en.user_name(),
                "Scheduled task completed",
                "Network interface eth0 up",
                "Memory usage normal",
                "Backup completed successfully",
                "Certificate renewal check passed",
            ])
            lines.append(f"[{ts}] [{level}] {source}: {msg_text}")
        ext = random.choice([".log", ".log", ".txt"])
        fname = random.choice(["syslog", "app", "service", "update", "network",
                                "security", "backup", "cron", "install", "boot"])
        (logs_dir / f"{fname}_{i:03d}{ext}").write_text("\n".join(lines), encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Software docs (~10) ---
    docs_dir = base / "Work" / "Projects"
    for i in range(10):
        project_name = fake_en.bothify("????-####")
        content = f"# {project_name} README\n\n"
        content += f"## Overview\n{fake_en.paragraph(nb_sentences=3)}\n\n"
        content += f"## Installation\n```bash\nnpm install {project_name}\n```\n\n"
        content += f"## Usage\n{fake_en.paragraph(nb_sentences=2)}\n\n"
        content += f"## API Reference\n"
        for _ in range(random.randint(2, 5)):
            content += f"\n### `{fake_en.bothify('???_????')}()`\n{fake_en.sentence(nb_words=8)}\n"
        (docs_dir / f"readme_{project_name}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    return count


def _noise_digital_clutter(base: Path, progress, task) -> int:
    """Generate digital clutter: browser noise, app exports, duplicates, junk."""
    count = 0

    # --- Browser history noise (~5 JSON files with hundreds of mundane URLs) ---
    app_dir = base / "AppData"
    app_dir.mkdir(parents=True, exist_ok=True)
    mundane_sites = [
        "https://www.reddit.com/r/{sub}",
        "https://www.youtube.com/watch?v={vid}",
        "https://stackoverflow.com/questions/{qid}",
        "https://www.amazon.com/dp/{asin}",
        "https://news.ycombinator.com/item?id={hnid}",
        "https://www.nytimes.com/{year}/{month}/article.html",
        "https://www.weather.com/forecast/{city}",
        "https://www.espn.com/scores",
        "https://maps.google.com/search/{query}",
        "https://www.netflix.com/browse",
        "https://www.linkedin.com/feed/",
        "https://twitter.com/home",
        "https://www.wikipedia.org/wiki/{topic}",
        "https://github.com/{user}/{repo}",
        "https://docs.google.com/document/d/{docid}",
    ]
    subreddits = ["funny", "AskReddit", "pics", "gaming", "todayilearned", "aww",
                   "science", "worldnews", "movies", "DIY", "food", "sports"]
    for i in range(5):
        history = []
        for _ in range(random.randint(80, 200)):
            tmpl = random.choice(mundane_sites)
            url = tmpl.format(
                sub=random.choice(subreddits),
                vid=fake_en.bothify("???????????"),
                qid=random.randint(10000, 9999999),
                asin=fake_en.bothify("B0########"),
                hnid=random.randint(30000000, 40000000),
                year=random.randint(2022, 2025),
                month=f"{random.randint(1,12):02d}",
                city=fake_en.city().lower().replace(" ", "-"),
                query=fake_en.sentence(nb_words=2).replace(" ", "+").lower().rstrip("."),
                topic=random.choice(_WIKIPEDIA_TOPICS),
                user=fake_en.user_name(),
                repo=fake_en.bothify("????-####"),
                docid=fake_en.sha256()[:20],
            )
            history.append({
                "url": url,
                "title": random.choice([
                    fake_en.catch_phrase(), fake_en.sentence(nb_words=5),
                    f"r/{random.choice(subreddits)} - Reddit",
                    f"YouTube - {fake_en.catch_phrase()}",
                    f"Stack Overflow - {fake_en.sentence(nb_words=4)}",
                ]),
                "visited_at": _random_date().isoformat(),
            })
        (app_dir / f"browser_history_{i:03d}.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        count += 1
        progress.advance(task)

    # --- App data exports (~10 JSON): Uber, DoorDash, fitness ---
    for i in range(10):
        app_type = random.choice(["uber_rides", "food_delivery", "fitness_tracker",
                                   "music_history", "shopping_orders"])
        if app_type == "uber_rides":
            entries = [{"date": _random_date().isoformat(),
                        "pickup": fake_en.address().replace("\n", ", "),
                        "dropoff": fake_en.address().replace("\n", ", "),
                        "fare": round(random.uniform(8, 45), 2),
                        "rating": random.choice([4, 5, 5, 5, 3])}
                       for _ in range(random.randint(10, 30))]
        elif app_type == "food_delivery":
            entries = [{"date": _random_date().isoformat(),
                        "restaurant": fake_en.company(),
                        "items": [fake_en.sentence(nb_words=3) for _ in range(random.randint(1, 4))],
                        "total": round(random.uniform(12, 65), 2)}
                       for _ in range(random.randint(8, 25))]
        elif app_type == "fitness_tracker":
            entries = [{"date": _random_date().strftime("%Y-%m-%d"),
                        "steps": random.randint(2000, 15000),
                        "calories": random.randint(1500, 3000),
                        "active_minutes": random.randint(10, 90),
                        "heart_rate_avg": random.randint(60, 95)}
                       for _ in range(random.randint(20, 60))]
        elif app_type == "music_history":
            entries = [{"played_at": _random_date().isoformat(),
                        "track": fake_en.catch_phrase(),
                        "artist": fake_en.name(),
                        "duration_ms": random.randint(120000, 360000)}
                       for _ in range(random.randint(30, 100))]
        else:
            entries = [{"date": _random_date().isoformat(),
                        "item": fake_en.catch_phrase(),
                        "price": round(random.uniform(5, 200), 2),
                        "status": random.choice(["Delivered", "Delivered", "Returned", "In Transit"])}
                       for _ in range(random.randint(10, 30))]
        (app_dir / f"{app_type}_{i:03d}.json").write_text(
            json.dumps({"type": app_type, "entries": entries}, indent=2), encoding="utf-8"
        )
        count += 1
        progress.advance(task)

    # --- Duplicate / near-duplicate files (~20) ---
    dup_texts = []
    for _ in range(5):
        dup_texts.append(fake_en.paragraph(nb_sentences=8))
    personal_dir = base / "Personal" / "Photos"
    personal_dir.mkdir(parents=True, exist_ok=True)
    desktop_dir = base / "Desktop"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = base / "Downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        src_text = random.choice(dup_texts)
        suffix = random.choice(["_copy", " (2)", "_backup", "_old", " - Copy"])
        dest = random.choice([desktop_dir, dl_dir])
        if i < 10:
            fname = f"document{suffix}_{i:02d}.txt"
            dest_path = dest / fname
            dest_path.write_text(src_text, encoding="utf-8")
        else:
            label = f"Photo copy #{i}"
            img_bytes = _make_image(label, 320, 240, bg="#cccccc")
            fname = f"photo_{i:02d}{suffix.replace(' ', '_')}.jpg"
            dest_path = dest / fname
            dest_path.write_bytes(img_bytes)
        count += 1
        progress.advance(task)

    # --- Empty / corrupted files (~15) ---
    temp_dir = base / "Temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        ftype = random.choice(["empty", "truncated", "garbled"])
        ext = random.choice([".tmp", ".dat", ".csv", ".txt", ".bak"])
        fname = f"file_{i:03d}{ext}"
        dest = random.choice([temp_dir, dl_dir])
        if ftype == "empty":
            (dest / fname).write_bytes(b"")
        elif ftype == "truncated":
            partial = fake_en.paragraph(nb_sentences=2)[:random.randint(10, 50)]
            (dest / fname).write_text(partial, encoding="utf-8")
        else:
            garbled = bytes(random.randint(0, 255) for _ in range(random.randint(50, 500)))
            (dest / fname).write_bytes(garbled)
        count += 1
        progress.advance(task)

    # --- System / temp files (~10) ---
    sys_names = [
        "desktop.ini", "Thumbs.db", ".DS_Store", "~$document.docx",
        "swapfile.sys", "pagefile.sys", "ntuser.dat.LOG",
        "debug.log", "crash_report.txt", "update_cache.tmp",
    ]
    for fname in sys_names:
        content = fake_en.binary(length=random.randint(20, 200)) if "." not in fname[-4:] or fname.endswith(
            (".sys", ".db", ".dat", ".LOG")) else fake_en.paragraph(nb_sentences=1).encode()
        (temp_dir / fname).write_bytes(content)
        count += 1
        progress.advance(task)

    # --- Cached web pages (~10 HTML) ---
    cache_dir = base / "AppData" / "Cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        headline = random.choice(_NEWS_HEADLINES)
        lf = fake_en
        html = (
            f"<!DOCTYPE html>\n<html><head><title>{headline}</title></head>\n"
            f"<body>\n<h1>{headline}</h1>\n"
            f"<p class='date'>{_random_date().strftime('%B %d, %Y')}</p>\n"
            f"<p>{lf.paragraph(nb_sentences=5)}</p>\n"
            f"<p>{lf.paragraph(nb_sentences=4)}</p>\n"
            f"<p>{lf.paragraph(nb_sentences=6)}</p>\n"
            f"</body></html>"
        )
        (cache_dir / f"cached_page_{i:03d}.html").write_text(html, encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Download folder junk (~40) ---
    for i in range(40):
        ftype = random.choice(["readme", "tos", "changelog", "installer_readme", "release_notes"])
        lf = LOCALE_FAKERS[_pick_locale()]
        if ftype == "readme":
            content = f"README\n\n{lf.paragraph(nb_sentences=5)}\n\nVersion {random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}\n"
        elif ftype == "tos":
            content = f"TERMS OF SERVICE\n\nLast updated: {_random_date().strftime('%B %d, %Y')}\n\n"
            content += "\n\n".join(lf.paragraph(nb_sentences=random.randint(3, 6)) for _ in range(random.randint(5, 10)))
        elif ftype == "changelog":
            content = f"CHANGELOG\n\n"
            for v in range(random.randint(3, 8)):
                content += f"## v{random.randint(1,3)}.{v}.0 - {_random_date().strftime('%Y-%m-%d')}\n"
                content += f"- {lf.sentence(nb_words=6)}\n- {lf.sentence(nb_words=5)}\n\n"
        elif ftype == "installer_readme":
            content = f"Installation Guide\n\nSystem Requirements:\n- OS: Windows 10+ / macOS 12+\n"
            content += f"- RAM: {random.choice([4, 8, 16])}GB\n- Disk: {random.randint(1, 20)}GB\n\n"
            content += f"Steps:\n1. {lf.sentence(nb_words=5)}\n2. {lf.sentence(nb_words=6)}\n3. {lf.sentence(nb_words=4)}\n"
        else:
            content = f"Release Notes - v{random.randint(1,5)}.{random.randint(0,9)}\n\n"
            content += f"Date: {_random_date().strftime('%B %d, %Y')}\n\n"
            content += "\n".join(f"- {lf.sentence(nb_words=7)}" for _ in range(random.randint(5, 12)))
        ext = random.choice([".txt", ".txt", ".md", ".txt"])
        (dl_dir / f"{ftype}_{i:03d}{ext}").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    return count


def _noise_red_herrings(base: Path, progress, task) -> int:
    """Generate mildly suspicious but ultimately benign content."""
    count = 0

    work_dir = base / "Work" / "Projects"
    work_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = base / "Downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    personal_dir = base / "Personal"
    personal_dir.mkdir(parents=True, exist_ok=True)

    # --- VPN configs (~5) ---
    vpn_dir = base / "AppData"
    vpn_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        server = random.choice(["us-east", "eu-west", "ap-southeast", "us-west", "eu-central"])
        content = (
            f"# OpenVPN Configuration - Corporate Remote Access\n"
            f"# IT Department - Do not modify\n"
            f"client\n"
            f"dev tun\n"
            f"proto udp\n"
            f"remote vpn-{server}.acme-corp.com 1194\n"
            f"resolv-retry infinite\n"
            f"nobind\n"
            f"persist-key\n"
            f"persist-tun\n"
            f"ca ca.crt\n"
            f"cert client-{fake_en.user_name()}.crt\n"
            f"key client-{fake_en.user_name()}.key\n"
            f"cipher AES-256-GCM\n"
            f"verb 3\n"
        )
        (vpn_dir / f"vpn_{server}_{i:03d}.ovpn").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Encrypted-looking filenames (~5) ---
    for i in range(5):
        name_patterns = [
            f"backup_{random.randint(2022, 2025)}_enc.dat",
            f"archive_{fake_en.bothify('????')}_locked.bin",
            f"vault_{random.randint(1, 99):02d}.gpg",
            f"secure_notes_{fake_en.bothify('###')}.aes",
            f"private_key_backup.pem",
        ]
        fname = name_patterns[i]
        content = fake_en.paragraph(nb_sentences=random.randint(3, 8))
        (personal_dir / fname).write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Large cash amount references (~10) ---
    fin_personal = base / "Personal" / "Finance"
    fin_personal.mkdir(parents=True, exist_ok=True)
    budget_types = [
        ("Home Renovation Budget", [
            ("Kitchen remodel", 25000, 45000), ("Bathroom update", 8000, 20000),
            ("New roof", 10000, 25000), ("Landscaping", 3000, 12000),
            ("Flooring", 5000, 15000), ("HVAC replacement", 6000, 12000),
        ]),
        ("Wedding Budget", [
            ("Venue", 15000, 40000), ("Catering", 10000, 25000),
            ("Photography", 3000, 8000), ("Flowers", 2000, 6000),
            ("DJ / Band", 2000, 5000), ("Dress / Tux", 2000, 8000),
        ]),
        ("College Savings Plan", [
            ("Tuition Year 1", 30000, 60000), ("Room & Board", 10000, 18000),
            ("Books & Supplies", 1000, 3000), ("Transportation", 1000, 5000),
        ]),
    ]
    for i in range(10):
        if i < len(budget_types):
            title, items = budget_types[i]
        else:
            title = f"Personal Budget {random.randint(2022, 2025)}"
            items = [(fake_en.sentence(nb_words=3), 500, 5000)
                     for _ in range(random.randint(4, 8))]
        path = fin_personal / f"budget_{i:03d}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Estimated", "Actual", "Notes"])
            for cat, lo, hi in items:
                est = random.randint(lo, hi)
                actual = round(est * random.uniform(0.8, 1.3))
                writer.writerow([cat, f"${est:,}", f"${actual:,}",
                                 random.choice(["", "Paid", "Pending", "Got a quote"])])
        count += 1
        progress.advance(task)

    # --- Legit foreign business docs (~15) ---
    intl_dir = base / "Work" / "Projects"
    legit_locales = ["es", "de", "pt", "ja", "zh"]
    legit_companies = {
        "es": "Distribuidora El Sol S.A. de C.V.",
        "de": "Müller & Schmidt Maschinenbau GmbH",
        "pt": "Exportadora Tropical Ltda.",
        "ja": "鈴木電子部品株式会社",
        "zh": "上海绿色科技有限公司",
    }
    for i in range(15):
        locale = random.choice(legit_locales)
        lf = LOCALE_FAKERS[locale]
        company = legit_companies.get(locale, lf.company())
        content = f"{company}\n{'=' * 40}\n\n"
        content += f"Date: {_random_date().strftime('%B %d, %Y')}\n\n"
        for _ in range(random.randint(3, 6)):
            content += f"{lf.paragraph(nb_sentences=random.randint(2, 4))}\n\n"
        (intl_dir / f"intl_doc_{locale}_{i:03d}.txt").write_text(content, encoding="utf-8")
        count += 1
        progress.advance(task)

    # --- Crypto hobby files (~15) ---
    crypto_dir = base / "Personal" / "Crypto"
    crypto_dir.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        if i < 5:
            content = f"Mining Profitability Calculator\n\n"
            content += f"GPU: {random.choice(['RTX 4090', 'RTX 3080', 'RX 7900 XTX', 'RTX 4070 Ti'])}\n"
            content += f"Hash Rate: {random.randint(80, 150)} MH/s\n"
            content += f"Power: {random.randint(200, 350)}W\n"
            content += f"Electricity: ${random.uniform(0.08, 0.15):.2f}/kWh\n"
            content += f"Daily Revenue: ${random.uniform(1, 8):.2f}\n"
            content += f"Daily Cost: ${random.uniform(0.5, 3):.2f}\n"
            content += f"Daily Profit: ${random.uniform(0.5, 5):.2f}\n"
            content += f"\nConclusion: {'Profitable' if random.random() > 0.3 else 'Not worth it at current prices'}\n"
            (crypto_dir / f"mining_calc_{i:03d}.txt").write_text(content, encoding="utf-8")
        elif i < 10:
            content = f"r/CryptoCurrency - Discussion Thread\n\n"
            content += f"Posted by u/{fake_en.user_name()} · {random.randint(1, 23)}h\n\n"
            titles = [
                "Is DCA still the best strategy in this market?",
                "My 2-year crypto journey: lessons learned",
                "Should I stake my ETH or keep it liquid?",
                "Best hardware wallet for long-term storage?",
                "Tax implications of staking rewards - help!",
            ]
            content += f"{random.choice(titles)}\n\n"
            for _ in range(random.randint(3, 6)):
                content += f"u/{fake_en.user_name()}: {fake_en.sentence(nb_words=random.randint(8, 15))}\n\n"
            (crypto_dir / f"reddit_crypto_{i:03d}.txt").write_text(content, encoding="utf-8")
        else:
            portfolio = {
                "last_updated": _random_date().isoformat(),
                "holdings": [
                    {"coin": "BTC", "amount": round(random.uniform(0.001, 0.05), 6),
                     "avg_cost": random.randint(20000, 50000), "note": "Long term hold"},
                    {"coin": "ETH", "amount": round(random.uniform(0.01, 2), 4),
                     "avg_cost": random.randint(1000, 3000), "note": "Staking on Coinbase"},
                    {"coin": random.choice(["SOL", "ADA", "DOT"]),
                     "amount": round(random.uniform(1, 100), 2),
                     "avg_cost": round(random.uniform(5, 50), 2), "note": "Speculative"},
                ],
                "total_invested_usd": random.randint(500, 5000),
            }
            (crypto_dir / f"portfolio_{i:03d}.json").write_text(
                json.dumps(portfolio, indent=2), encoding="utf-8"
            )
        count += 1
        progress.advance(task)

    return count


def generate_noise(base: Path, progress, task) -> int:
    """Generate ~500 noise files to create a realistic signal-to-noise ratio."""
    count = 0
    count += _noise_downloaded_media(base, progress, task)
    count += _noise_personal(base, progress, task)
    count += _noise_work(base, progress, task)
    count += _noise_digital_clutter(base, progress, task)
    count += _noise_red_herrings(base, progress, task)
    return count


def generate_sample_drive(
    output_dir: Path | None = None,
    *,
    skip_noise: bool = False,
    scenario: str = "default",
) -> Path:
    """Generate the complete synthetic drive structure.

    Args:
        output_dir: Target directory (defaults to sample_drive/).
        skip_noise: Skip noise file generation.
        scenario: "default" for the 3-thread demo, "workshop" for the
                  single-thread Instruqt workshop scenario.
    """
    base = output_dir or SAMPLE_DRIVE_DIR
    if base.exists():
        import shutil
        shutil.rmtree(base)
    base.mkdir(parents=True)

    is_workshop = scenario == "workshop"
    evidence_estimated = 20 if is_workshop else 23
    noise_estimated = 0 if skip_noise else 500
    total_estimated = 32 + 39 + 37 + 23 + 15 + evidence_estimated + noise_estimated

    with Progress() as progress:
        task = progress.add_task("Generating multilingual sample data...", total=total_estimated)

        counts = {
            "Financial": generate_financial(base, progress, task),
            "Communications": generate_communications(base, progress, task),
            "Media": generate_media(base, progress, task),
            "Business Records": generate_business_records(base, progress, task),
            "Digital Artifacts": generate_digital_artifacts(base, progress, task),
        }

        if is_workshop:
            counts["Evidence Threads"] = generate_workshop_evidence(base, progress, task)
        else:
            counts["Evidence Threads"] = generate_evidence_threads(base, progress, task)

        if not skip_noise:
            counts["Noise"] = generate_noise(base, progress, task)

    total = sum(counts.values())
    signal = total - counts.get("Noise", 0)
    noise = counts.get("Noise", 0)
    print(f"\nGenerated {total} files in {base}")
    for cat, n in counts.items():
        print(f"  {cat}: {n} files")
    print(f"\n  Languages: {', '.join(LOCALES)}")
    if is_workshop:
        print("  Scenario: workshop (sanctions evasion)")
    else:
        print("  Evidence threads: Operation Oceanic, Operation Alpine, Operation Silk Road")
    if noise:
        print(f"  Signal-to-noise ratio: ~{signal}:{noise} ({signal / total * 100:.0f}% signal)")

    return base
