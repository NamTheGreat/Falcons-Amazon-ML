import pandas as pd, numpy as np, collections
D='student_resource/dataset/train/'
gt=pd.read_csv(D+'train_ground_truth.tsv',sep='\t',dtype=str,keep_default_na=False)
ids=[x for l in gt.matched_entity_ids for x in (l.split(',') if l else [])]
c=collections.Counter(ids)
print('matched ids',len(ids),'unique',len(c),'max multiplicity',max(c.values()))
s2=pd.read_csv(D+'train_source2.tsv',sep='\t',dtype=str,keep_default_na=False)
s3=pd.read_csv(D+'train_source3.tsv',sep='\t',dtype=str,keep_default_na=False)
print('s2',len(s2),'matched',s2.entity_id.isin(c).sum(),'s3',len(s3),'matched',s3.entity_id.isin(c).sum())
n2=[sum(1 for x in l.split(',') if x.startswith('S2')) if l else 0 for l in gt.matched_entity_ids]
n3=[sum(1 for x in l.split(',') if x.startswith('S3')) if l else 0 for l in gt.matched_entity_ids]
print(pd.crosstab(np.array(n2),np.array(n3)))
# unmatched samples
um=s2[~s2.entity_id.isin(c)].sample(15,random_state=0)
print(um.to_string())
um=s3[~s3.entity_id.isin(c)].sample(15,random_state=0)
print(um.to_string())
