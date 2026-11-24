"""User-approved sourced listing-rate overlay. No fitting, feature selection or raw writes."""
import csv,json,hashlib
from decimal import Decimal,InvalidOperation
from pathlib import Path
import pandas as pd
RAW_SHA256='55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d'
def rate_cents(x):
 try:d=Decimal(str(x))
 except InvalidOperation:raise ValueError('Invalid rate')
 if not d.is_finite() or not 0<d<100 or d*100!=(d*100).to_integral_value():raise ValueError('Rate must be percentage units with at most 2 decimals')
 return int(d*100)
def load_approved_ledger(directory):
 directory=Path(directory).resolve()
 policy=json.loads((directory/'approved_rate_rule_v1.json').read_text(encoding='utf-8-sig'))
 if not policy.get('adopted') or policy.get('decision_id')!='G2-RATE-2026-11-24-v1':raise ValueError('Approved policy required')
 path=(directory/policy['ledger']).resolve()
 if not path.is_relative_to(directory):raise ValueError('Ledger must be inside approved policy folder')
 if hashlib.sha256(path.read_bytes()).hexdigest()!=policy['ledger_sha256']:raise ValueError('Approved ledger hash changed')
 with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 ledger={}
 for r in rows:
  ident=r['id']
  if not ident.isdigit() or ident in ledger:raise ValueError('Invalid or duplicate ledger ID')
  if not r['listing_source_url'].startswith('https://www.sec.gov/Archives/edgar/data/1409970/'):raise ValueError('Missing posting source')
  old,new=rate_cents(r['expected_raw_rate_percent']),rate_cents(r['listing_rate_percent'])
  if old==new:raise ValueError('No-op ledger entry')
  ledger[ident]={'raw_cents':old,'listing_cents':new,'source_url':r['listing_source_url']}
 if len(ledger)!=148:raise ValueError('Expected approved 148-case ledger')
 return ledger
def apply_listing_rates(frame,ledger):
 """Return a corrected copy and separate audit rows. Does not add audit fields to model data."""
 if not {'id','int_rate'}.issubset(frame.columns):raise ValueError('id/int_rate required')
 if frame.id.astype(str).duplicated().any():raise ValueError('Duplicate IDs in input chunk')
 out=frame.copy(deep=True);audit=[];numeric=pd.api.types.is_numeric_dtype(frame.int_rate)
 for idx in frame.index:
  ident=str(frame.at[idx,'id'])
  if ident not in ledger:continue
  entry=ledger[ident];actual=rate_cents(frame.at[idx,'int_rate'])
  if actual!=entry['raw_cents']:raise ValueError('Stored rate drift or repeated overlay for '+ident)
  new=entry['listing_cents']/100
  out.at[idx,'int_rate']=new if numeric else f'{new:.2f}'
  audit.append({'id':ident,'int_rate_raw_percent':actual/100,'int_rate_working_percent':new,'source_url':entry['source_url']})
 if not frame.drop(columns='int_rate').equals(out.drop(columns='int_rate')):raise AssertionError('Other fields changed')
 if list(frame.columns)!=list(out.columns) or not frame.index.equals(out.index):raise AssertionError('Schema/order changed')
 return out,audit
def verify_coverage(seen,ledger):
 if set(seen)!=set(ledger):raise ValueError('Not all approved cases found in eligible input')
def verify_raw_fingerprint(path):
 with Path(path).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=RAW_SHA256:raise ValueError('Unexpected raw input fingerprint')
 return actual

