"""Build a stratified 60-image sample (+20 reserves) of MAP2 neuron images from idr0173 cingulate cortex."""
import pandas as pd, numpy as np
# Run from a clone of https://github.com/IDR/idr0173-breiter-alphasynuclein
R='repo/experimentA/'
a=pd.read_csv(R+'idr0173-experimentA-annotation.csv',encoding='utf-8-sig')
p=pd.read_csv(R+'idr0173-experimentA-processed.csv',encoding='utf-8-sig')
fp=pd.read_csv(R+'idr0173-experimentA-filePaths.tsv',sep='\t',header=None,names=['ds','url','img'])
fp['Image File']=fp.url.str.split('S-BIAD2216/').str[1]
d=(a.merge(p[['Image File','Oligomer Count','Oligomer Intensity']],on='Image File',how='left')
    .merge(fp[['Image File','url']],on='Image File',how='left'))
assert len(d)==len(a)
d=d[(d['Characteristics [Cell Type]']=='neuron')&(d['Characteristics [Brain Region]']=='cingulate')].copy()
print('cingulate MAP2 images:',len(d),'| with oligomer counts:',d['Oligomer Count'].notna().sum())
key='Oligomer Count' if d['Oligomer Count'].notna().all() else None
rng=np.random.default_rng(173); sel=[]
for grp,n_extra in [('HC',4),('PD',2)]:
    g=d[d['Characteristics [Disease State]']==grp]
    for pid,gp in g.groupby('Characteristics [Patient ID]'):
        if key:  # one image from the lower half and one from the upper half of this patient's oligomer counts
            med=gp[key].median(); parts=[gp[gp[key]<=med],gp[gp[key]>med]]
        else: parts=[gp,gp]
        for part in parts:
            pool=(part if len(part) else gp).drop(index=sel,errors='ignore')
            sel.append(pool.sample(1,random_state=int(rng.integers(1e9))).index[0])
    sel+=list(g.drop(index=sel,errors='ignore').sample(n_extra,random_state=173).index)
S=d.loc[sel].copy(); S['set']='main'
Rv=d.drop(index=sel).groupby('Characteristics [Disease State]').sample(10,random_state=7).copy(); Rv['set']='reserve'
out=pd.concat([S,Rv]).rename(columns={'Image Name':'image_name','Characteristics [Disease State]':'disease',
 'Characteristics [Patient ID]':'patient','Characteristics [Sex]':'sex','Characteristics [Age of Death]':'age_death',
 'Characteristics [Alpha-syn Braak Stage]':'braak','Characteristics [Disease Duration (yrs)]':'pd_duration_yrs',
 'Characteristics [PMI]':'pmi','Oligomer Count':'oligomer_count','Oligomer Intensity':'oligomer_intensity','Channels':'channels'})
out.insert(0,'sample_id',[f'S{i:02d}' for i in range(1,61)]+[f'R{i:02d}' for i in range(1,21)])
cols=['sample_id','set','disease','patient','sex','age_death','pmi','braak','pd_duration_yrs','oligomer_count','oligomer_intensity','channels','image_name','Image File','url']
out[cols].rename(columns={'Image File':'image_path'}).to_csv('out/idr0173_sample_manifest.csv',index=False)
m=out[out.set=='main']
print(m.groupby('disease').size().to_dict(),'patients:',m.patient.nunique())
print(m.groupby('disease')['oligomer_count'].describe().round(0).to_string())