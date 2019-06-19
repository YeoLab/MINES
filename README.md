# MINES
(m)6A (I)dentification Using (N)anopor(E) (S)equencing



Tombo(v1.4) Commands Prior to MINES:

(Only required if fast5s do not already contain fastqs)
tombo preprocess annotate_raw_with_fastqs --fast5-basedir /fast5_dir/ --fastq-filenames Fastqs --overwrite --processes #

tombo resquiggle /fast5_dir/ REF.fa --overwrite --processes #

tombo detect_modifications de_novo --fast5-basedirs /fast5_dir/ --statistics-file-basename Stats_Filename

tombo text_output browser_files --fast5-basedirs /fast5_dir/ --statistics-filename Stats_Filename.tombo.stats --browser-file-basename output_filename --file-types coverage fraction
