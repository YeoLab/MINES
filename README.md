# MINES
(m)6A (I)dentification Using (N)anopor(E) (S)equencing



## Tombo(v1.4) Commands Prior to MINES:

(Only required if fast5s do not already contain fastqs)

tombo preprocess annotate_raw_with_fastqs --fast5-basedir /fast5_dir/ --fastq-filenames Fastqs --overwrite --processes #

tombo resquiggle /fast5_dir/ REF.fa --overwrite --processes #

tombo detect_modifications de_novo --fast5-basedirs /fast5_dir/ --statistics-file-basename Stats_Filename

tombo text_output browser_files --fast5-basedirs /fast5_dir/ --statistics-filename Stats_Filename.tombo.stats --browser-file-basename output_filename --file-types coverage fraction

## MINES Dependencies:
python                    3.7.0  
pandas                    0.23.4  
pybedtools                0.8.0  
numpy*                     1.16.4  
scikit-learn              0.19.2  
bedops                     2.4.35  

## MINES Usage
### Convert Wig Files Into Bed Files.
wig2bed < output_filename.fraction_modified_reads.plus.wig > output_filename.fraction_modified_reads.plus.wig.bed

### cDNA_MINES Example:
python cDNA_MINES.py --fraction_modified output_filename.fraction_modified_reads.plus.wig.bed --coverage output_filename.coverage.plus.bedgraph --output m6A_output_filename.bed --ref REF.fa

### genomic_MINES Example:
python genomic_MINES.py --fraction_modified_plus output_filename.fraction_modified_reads.plus.wig.bed --coverage_plus output_filename.coverage.plus.bedgraph --coverage_minus output_filename.coverage.minus.bedgraph --fraction_modified_minus output_filename.fraction_modified_reads.minus.wig.bed --output m6A_output_filename.bed --ref REF.fa




### Output File Format(bed/tab delimited):
chr,   start,   stop,    5-mer,   unique key,    strand,    fraction modified^,    coverage  
^fraction modified is the value at the identified m6A site. However, the value at this position should be used with caution as the "A" site was found to be a poor predictor of methylation.











## Reference files can be downloaded from:
hg19:http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/ File:hg19.fa.gz then run gunzip.  
cDNA:ftp://ftp.ensembl.org/pub/release-91/fasta/homo_sapiens/cdna/  File:Homo_sapiens.GRCh38.cdna.all.fa.gz then run gunzip.
