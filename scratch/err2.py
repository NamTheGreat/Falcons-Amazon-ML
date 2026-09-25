import polars as pl
W='work/'
v=pl.read_parquet(W+'val_frame.parquet')
thr=0.68
top=v.sort('p2',descending=True).unique('qid',keep='first')
qpos=v.group_by('qid').agg((pl.col('y').max()==1).alias('has_pos'))
top=top.join(qpos,on='qid')
tot_pos_frame=v['y'].sum()
print('pos in frame',tot_pos_frame)
print('FP:', top.filter((pl.col('p2')>=thr)&(pl.col('y')==0)).height, ' (query has other true:', top.filter((pl.col('p2')>=thr)&(pl.col('y')==0)&pl.col('has_pos')).height,')')
print('FN top true below thr:', top.filter((pl.col('p2')<thr)&(pl.col('y')==1)).height)
print('FN wrong top:', top.filter((pl.col('y')==0)&pl.col('has_pos')).height)
print('TP:', top.filter((pl.col('p2')>=thr)&(pl.col('y')==1)).height)
# by source/nonlatin/has addr
top=top.with_columns((pl.col('q_alen')>0).alias('has_addr'))
print(top.filter(pl.col('has_pos')).group_by('q_nonlatin','has_addr').agg(pl.len(),((pl.col('p2')>=thr)&(pl.col('y')==1)).mean().alias('tp_rate')).sort('len'))
def show(df,n=25,seed=0):
    df=df.sample(min(n,df.height),seed=seed)
    qs=pl.concat([pl.scan_parquet(W+'train_s2.parquet'),pl.scan_parquet(W+'train_s3.parquet')]).with_row_index('qid').select('qid','raw_name','raw_addr').filter(pl.col('qid').is_in(df['qid'].implode())).collect()
    ss=pl.scan_parquet(W+'train_s1.parquet').with_row_index('sid').select('sid',pl.col('raw_name').alias('s_name'),pl.col('raw_addr').alias('s_addr')).filter(pl.col('sid').is_in(df['sid'].implode())).collect()
    out=df.join(qs,on='qid').join(ss,on='sid')
    for r in out.iter_rows(named=True):
        print(f"p={r['p2']:.2f} p1={r['p1']:.2f} y={r['y']} | {r['raw_name']} | {r['raw_addr']}\n            S1: {r['s_name']} | {r['s_addr']}")
print('\n=== FN true top below thr'); show(top.filter((pl.col('p2')<thr)&(pl.col('y')==1)),30,3)
