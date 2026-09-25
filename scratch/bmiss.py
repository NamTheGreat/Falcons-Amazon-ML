import sys, polars as pl
sys.path.insert(0,'code/business_entity_resolution/src')
from train import load_gt_pairs
from run_blocking import load_split
s1,q=load_split('work','train',['entity_id','has_addr','nonlatin_name','name_alt','raw_name','raw_addr','country'])
gt=load_gt_pairs('student_resource/dataset/train/train_ground_truth.tsv',s1,q)
c=pl.read_parquet('work/train_cands.parquet',columns=['qid','sid'])
m=gt.join(c.with_columns(pl.lit(True).alias('hit')),on=['qid','sid'],how='left').with_columns(pl.col('hit').fill_null(False))
m=m.join(q.select(pl.col('rid').alias('qid'),'has_addr','nonlatin_name',(pl.col('name_alt')!='').alias('dba'),'country'),on='qid')
print('recall',m['hit'].mean(), 'misses', (~m['hit']).sum())
print(m.group_by('has_addr','nonlatin_name').agg(pl.len(),pl.col('hit').mean(),(~pl.col('hit')).sum().alias('miss')).sort('miss'))
# was the true S1 in candidates of query at all? query got any cands?
qc=c.group_by('qid').agg(pl.len().alias('nc'))
mm=m.filter(~pl.col('hit')).join(qc,on='qid',how='left').with_columns(pl.col('nc').fill_null(0))
print(mm.group_by('nc').agg(pl.len()).sort('nc'))
mm=mm.filter(pl.col('has_addr')).sample(40,seed=7).join(q.select(pl.col('rid').alias('qid'),'raw_name','raw_addr'),on='qid').join(s1.select(pl.col('rid').alias('sid'),pl.col('raw_name').alias('sn'),pl.col('raw_addr').alias('sa')),on='sid')
for r in mm.iter_rows(named=True): print(f"{r['raw_name']} | {r['raw_addr']}\n      S1: {r['sn']} | {r['sa']}")
