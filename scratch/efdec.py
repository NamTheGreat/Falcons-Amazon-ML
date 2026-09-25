import sys, time, polars as pl
sys.path.insert(0,'code/business_entity_resolution/src')
from evaluate import decide, decide_expected_f, macro_f05
from train import load_gt_pairs
from run_blocking import load_split
s1,q=load_split('work','train',['entity_id'])
gt=load_gt_pairs('student_resource/dataset/train/train_ground_truth.tsv',s1,q)
hold=s1.select(pl.col('rid').alias('sid'),(pl.col('entity_id').hash(seed=11)%50==0).alias('hold')).filter('hold')
gth=gt.join(hold,on='sid',how='semi'); hs=hold['sid']
v=pl.read_parquet('work/val_frame.parquet',columns=['qid','sid','p2'])
print('global thr .68', macro_f05(decide(v,0.68),gth,hs))
for mr in [0.0,0.1,0.3]:
    t=time.time(); r=macro_f05(decide_expected_f(v,miss_rate=mr),gth,hs); print('EF miss',mr,r,round(time.time()-t,1),'s')
for a in [0.8,1.2,1.5]:
    vv=v.with_columns(pl.col('p2')**a)
    print('EF pow',a,macro_f05(decide_expected_f(vv),gth,hs))
