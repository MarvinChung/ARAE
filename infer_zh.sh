#python train.py --data_path ./data_zh --load_vocab ./data_zh/word_id_50000.json --vocab_size 50000 --outf zh_example
python infer.py --data_path ./data_zh/seq2seq_zh.txt --outf zh_out --corpus_name zh
