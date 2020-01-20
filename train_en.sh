outf=en_out
mkdir -p en_out
python train.py --data_path ./data_en --vocab_size 50000 --outf $outf 
