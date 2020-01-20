outf=zh_out
mkdir -p zh_out
python train.py --data_path ./data_zh --vocab_size 50000 --outf $outf 
