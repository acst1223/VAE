from pipeline import csv_extractor, csv_block_extractor, splitter

origin_log_file = 'HDFS/HDFS.log'
col_header_file = 'HDFS/col_header.txt'
label_file = 'HDFS/label.csv'
train_ratio = 0.6
test_ratio = 0.3
tot_syms = 29

csv_log_file = origin_log_file[: -4] + '.csv'
csv_extracted_file = csv_log_file[: -4] + '_extracted.csv'
train_file = csv_log_file[: -4] + '_train.csv'
validate_file = csv_log_file[: -4] + '_validate.csv'
test_file = csv_log_file[: -4] + '_test.csv'

csv_block_extractor.load_label_file(label_file)

# STEP 1
csv_extractor.csv_extracting(origin_log_file, col_header_file, label_file)
