from __future__ import division
import argparse, os
import pandas as pd
from sklearn.externals import joblib
import pybedtools
import numpy as np

def __is_valid_file(arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        # File exists so return the filename
        return arg

# script arguments
parser = argparse.ArgumentParser(description='MINES(M6A Identification using NanoporE Sequencing): Takes Tombo coverage and fraction modified values and returns m6A predictions for AGACT, GGACA, GGACC, and GGACT sequences')
parser.add_argument('--ref',dest='ref', type=__is_valid_file,required=True, help='Reference Genome Fasta File')
parser.add_argument('--coverage',dest='coverage', type=__is_valid_file, required=True, help='Tombo Text_Output Coverage File(*plus.bedgraph)')
parser.add_argument('--fraction_modified',dest='fraction_modified', type=__is_valid_file, required=True, help='Tombo Text_Output Fraction_Modified File (*plus.wig)')
parser.add_argument('--output',dest='output', required=True, help='Output Filename')

args = parser.parse_args()

coverage=args.coverage
ref=args.ref
f_modified=args.fraction_modified
output=args.output

# Search for DRACH motifs in data
drach_seqs=['AGACT','GGACA','GGACC','GGACT']
tmp=pybedtools.BedTool(coverage)
tmp=tmp.to_dataframe(header=None, names=['chr','start','stop','coverage','5','strand'])
tmp=tmp[tmp['coverage']>4]
tmp['5']=0
tmp['strand']='+'

tmp['start']=tmp['start'].astype(int)
tmp['stop']=tmp['stop'].astype(int)
tmp=pybedtools.BedTool.from_dataframe(tmp)
tmp=tmp.merge()
seqs=tmp.sequence(fi=ref, s=True)
ref=None
x = open(seqs.seqfn).read()
d = dict()
for i in x.split('\n'):
    if '>' in i:
        name = i
    else:
        d[name] = i
name=None
seqs=None
seqs_df = pd.DataFrame.from_dict(d, orient='index')
seqs_df.columns=['seq']
seqs_df['start'] = list(map(lambda x: x.split(':')[1].split('-')[0], seqs_df.index))
seqs_df['stop'] = list(map(lambda x: x.split(':')[1].split('-')[1].split('(')[0], seqs_df.index))
seqs_df['chrom'] = list(map(lambda x: x.split('>')[1].split(':')[0], seqs_df.index))
d = dict()
l1 = list()
l2 = list()
l3 = list()
l4 = list()
l5 = list()
l6 = list()
for i in seqs_df.index:
    l = list()
    s = seqs_df.loc[i,'seq']
    start = int(seqs_df.loc[i,'start'])
    chrom = seqs_df.loc[i,'chrom']
    for j in range(0, len(s)-4):
        if s[j:j+5].upper() in drach_seqs:
            pos = start+j+3
            window = range(pos-15, pos+15)
            window_stop = [a + 1 for a in window]
            pos_within_window = [a - pos for a in window]
            l1.extend([chrom]*len(window))
            l2.extend(window)
            l3.extend(window_stop)
            l4.extend([pos]*len(window))
            l5.extend([s[j:j+5].upper()]*len(window))
            l6.extend(pos_within_window)
df = pd.DataFrame({'chr':l1, 'start':l2,'stop':l3,'drach_coordinate':l4,'drach_kmer':l5,'pos':l6})
df['strand']='+'
i=None
seqs_df = None
d = None
l1 = None
l2 = None
l3 = None
l4 = None
l5 = None
l6 = None
l = None
s = None
start = None
chrom = None
pos = None
window = None
window_stop = None
pos_within_window = None


df['key2'] = df['chr'] + ':' + df['start'].astype(str)
df = df.set_index('key2')

#Assign fraction modified values
fraction_modified= pd.read_csv(f_modified, sep='\t', header=None)
fraction_modified['key'] = fraction_modified[0] + ':' + fraction_modified[1].astype(str)
fraction_modified = fraction_modified.set_index('key')
s = set(df.index).intersection(set(fraction_modified.index))

fraction_modified = fraction_modified.loc[s]
df = df.join(fraction_modified[[4]])
fraction_modified=None
f_modified= None
s=None

df=df[df['start']>0]


#Assign coverage values
df=pybedtools.BedTool.from_dataframe(df=df)
tmp=pybedtools.BedTool(coverage)
coverage=None
df=df.intersect(tmp, wb=True, wa=True)
tmp=None
df=df.to_dataframe(header=None, names=['chr','start','stop','drach_coordinate','drach_kmer','pos','strand','f_mod','chr.2','start.2','stop.2','coverage'])
df=df[['chr','start','stop','drach_coordinate','drach_kmer','strand','pos','f_mod','coverage']]
df=df[df['coverage']>=5]
df['key'] = df['chr'] + ':' + df['drach_coordinate'].astype(str) + ':' + df['drach_kmer']+ ':' +  df['strand']

#Pivot data for model
df=df.drop_duplicates(['key','pos'])
df_final = df.pivot(values='f_mod', index='key', columns='pos')
df_final = df_final.drop(columns=[-15,-14,-13,-12,-11,11,12,13,14])
df_final=df_final.replace('.', np.nan)
df_final = df_final.dropna()
df_final['kmer']=list(map(lambda x:x.split(':')[2], df_final.index))

#Load models
model_list = pd.read_csv('/mnt/c/Users/yeolab/Documents/m6A_paper_final_model_and_data/Models/names.txt', header=None, names=['file'])
model_list['kmer']=list(map(lambda x:x.split('_')[0], model_list.file))
model_list.set_index('kmer',inplace=True)

#Model predictions
preds=pd.DataFrame()
for kmer in drach_seqs:
    kmer_df=df_final[df_final['kmer']==kmer]
    kmer_df=kmer_df.drop(columns=['kmer'])
    fname=model_list.loc[kmer,'file']
    loaded_model = joblib.load('/mnt/c/Users/yeolab/Documents/m6A_paper_final_model_and_data/Models/'+fname)
    p=loaded_model.predict(kmer_df)
    kmer_df['pred']=p
    kmer_df['key']=kmer_df.index
    tmp=kmer_df[['key','pred']]
    preds=pd.concat([preds,tmp])

drach_seqs=None
kmer=None
model_list=None
df_final = None
kmer_df=None
fname=None
loaded_model = None
p=None
tmp=None

#Map predictions to initial data
mapping = dict(preds[['key','pred']].values)
df['pred'] = df.key.map(mapping)
preds=None
mapping=None

#Generate final bed file
df=df[(df['pos']==0)&(df['pred']==1.0)&(df['drach_kmer'].isin(['GGACT', 'GGACA', 'AGACT', 'GGACC']))&(df['coverage']>=5)]
df.to_csv(output,sep='\t', header=None, index=None, columns=['chr','start','stop','drach_kmer','key','strand','f_mod','coverage'])