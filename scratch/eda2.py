import pandas as pd
D='student_resource/dataset/train/'
r=lambda f: pd.read_csv(D+f,sep='\t',dtype=str,keep_default_na=False)
s1=r('train_source1.tsv').set_index('entity_id'); s23=pd.concat([r('train_source2.tsv'),r('train_source3.tsv')]).set_index('entity_id')
gt=r('train_ground_truth.tsv').sample(14,random_state=3)
for _,g in gt.iterrows():
    a=s1.loc[g.source1_entity_id]; print(f"\nS1 {a.business_name} | {a.business_address} | {a.country}")
    for m in (g.matched_entity_ids.split(',') if g.matched_entity_ids else []):
        b=s23.loc[m]; print(f"   {m[:2]} {b.business_name} | {b.business_address}")
