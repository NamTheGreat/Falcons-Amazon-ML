import sys, time, polars as pl
sys.path.insert(0,'code/business_entity_resolution/src')
from blocking import generate_candidates
W='work/'
s1=pl.read_parquet(W+'train_s1.parquet').with_row_index('rid')
q=pl.concat([pl.read_parquet(W+'train_s2.parquet'),pl.read_parquet(W+'train_s3.parquet')]).with_row_index('rid')
q=q.sample(int(sys.argv[1]) if len(sys.argv)>1 else 400000, seed=0)
gt=pl.read_csv('student_resource/dataset/train/train_ground_truth.tsv',separator='\t',infer_schema=False,missing_utf8_is_empty_string=True)
gt=gt.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids')!='')
gt=gt.join(s1.select(pl.col('entity_id').alias('source1_entity_id'),pl.col('rid').alias('sid')),on='source1_entity_id').join(q.select(pl.col('entity_id').alias('matched_entity_ids'),pl.col('rid').alias('qid')),on='matched_entity_ids')
print('gt pairs in sample',gt.height,'of queries',q.height)
t=time.time()
cols=['rid','country','name_core','name_alt','name_compact','name_skel','addr','addr_nums','nonlatin_name']
c=generate_candidates(s1.select(cols),q.select(cols),chunk=100000)
print('time',time.time()-t,'cands',c.height, 'per query',c.height/q.height)
m=gt.join(c,on=['qid','sid'],how='left')
print('recall', m['brank'].is_not_null().mean())
for k in [1,2,3,5,8,12]: print(k, (m['brank']<=k).mean())
miss=m.filter(pl.col('brank').is_null()).join(q.select(pl.col('rid').alias('qid'),'raw_name','raw_addr'),on='qid').join(s1.select(pl.col('rid').alias('sid'),pl.col('raw_name').alias('s1n'),pl.col('raw_addr').alias('s1a')),on='sid')
pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_rows(40)
print(miss.select('raw_name','raw_addr','s1n','s1a').sample(min(30,miss.height),seed=1))
c.write_parquet('work/_exp_c.parquet'); gt.write_parquet('work/_exp_gt.parquet')
