import sys, polars as pl, itertools
sys.path.insert(0,'code/business_entity_resolution/src')
from evaluate import macro_f05
from train import load_gt_pairs
from run_blocking import load_split
s1,q=load_split('work','train',['entity_id'])
gt=load_gt_pairs('student_resource/dataset/train/train_ground_truth.tsv',s1,q)
hold=s1.select(pl.col('rid').alias('sid'),(pl.col('entity_id').hash(seed=11)%50==0).alias('hold')).filter('hold')
gth=gt.join(hold,on='sid',how='semi'); hs=hold['sid']
v=pl.read_parquet('work/val_frame.parquet',columns=['qid','sid','p2'])
b=v.sort('p2',descending=True).unique('qid',keep='first')
b=b.with_columns(((pl.col('p2')>=0.9).sum().over('sid')-(pl.col('p2')>=0.9).cast(pl.Int32)).alias('nconf'))
for t0,t1,t2 in itertools.product([0.55,0.6,0.65,0.7],[0.6,0.68,0.75,0.8],[0.7,0.8,0.85]):
    thr=pl.when(pl.col('nconf')==0).then(t0).when(pl.col('nconf')<=2).then(t1).otherwise(t2)
    r=macro_f05(b.filter(pl.col('p2')>=thr).select('qid','sid'),gth,hs)
    print(t0,t1,t2,r)
