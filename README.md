# MINES
(m)6A (I)dentification Using (N)anopor(E) (S)equencing



Tombo(v1.4) Commands Prior to MINES:

(Only required if fast5s do not already contain fastqs)
tombo preprocess annotate_raw_with_fastqs --fast5-basedir /fast5_dir/ --fastq-filenames Fastqs --overwrite --processes #

tombo resquiggle /fast5_dir/ REF.fa --overwrite --processes #

tombo detect_modifications de_novo --fast5-basedirs /fast5_dir/ --statistics-file-basename Stats_Filename

tombo text_output browser_files --fast5-basedirs /fast5_dir/ --statistics-filename Stats_Filename.tombo.stats --browser-file-basename output_filename --file-types coverage fraction

MINES dependencies:
python                    3.7.0
pandas                    0.23.4
pybedtools                0.8.0
numpy*                     1.16.4
scikit-learn              0.19.2


genomic_MINES commands:

python cDNA_MINES.py --fraction_modified output_filename.fraction_modified_reads.plus.wig --coverage output_filename.coverage.plus.bedgraph --output m6A_output_filename.bed --ref REF.fa

python genomic_MINES.py --fraction_modified_plus output_filename.fraction_modified_reads.plus.wig --coverage_plus output_filename.coverage.plus.bedgraph --coverage_minus output_filename.coverage.minus.bedgraph --fraction_modified_minus output_filename.fraction_modified_reads.minus.wig --output m6A_output_filename.bed --ref REF.fa
