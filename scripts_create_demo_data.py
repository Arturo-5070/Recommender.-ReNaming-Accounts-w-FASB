from pathlib import Path
import pandas as pd

base = Path('/home/ubuntu/word_recommender/data')
base.mkdir(parents=True, exist_ok=True)

balances = pd.DataFrame([
    ['1000', 'Asset', '1250000', '2026-01', '  Cash   at Bank  '],
    ['1100', 'Asset', '540000', '2026-01', 'Trade Debtors'],
    ['1200', 'Asset', '80000', '2026-01', 'Raw   Materials Inventory'],
    ['2000', 'Liability', '-315000', '2026-01', 'Vendor   Payables'],
    ['2100', 'Liability', '-250000', '2026-01', 'Unearned Subscription Income'],
    ['3000', 'Equity', '-450000', '2026-01', 'Accumulated Earnings'],
    ['4000', 'Revenue', '-980000', '2026-01', 'Subscription Sales'],
    ['5000', 'Expense', '410000', '2026-01', 'Product   Development Expense'],
], columns=['account number', 'account type', 'balances', 'financial period', 'account description'])
balances.to_csv(base / 'demo_balances_utf16.csv', index=False, encoding='utf-16')

journals = pd.DataFrame([
    ['JE-0001', '2026-01-03', '1000', '150000', '2026-01'],
    ['JE-0002', '2026-01-04', '1100', '25000', '2026-01'],
    ['JE-0003', '2026-01-05', '2100', '40000', '2026-01'],
], columns=['document number', 'posting date', 'account number', 'amount', 'financial period'])
journals.to_csv(base / 'demo_journal_entries_utf16.csv', index=False, encoding='utf-16')

auditor = pd.DataFrame([
    ['1000', 'Cash and Cash Equivalents'],
    ['1100', 'Accounts Receivable, Net'],
    ['2100', 'Deferred Revenue'],
], columns=['account number', 'auditor approved label'])
auditor.to_csv(base / 'demo_auditor_labels_utf16.csv', index=False, encoding='utf-16')
