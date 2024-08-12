#for dataset in tweet
#do
#  for model in RWKV
#  do
#    python main.py --mode T --config_path configs/${model}-${dataset}.ini --out_dir /data/lastness/output/${model}-${dataset}
#    done
#  done
#
#for dataset in sina
#do
#  for model in RNN RWKV
#  do
#    python main.py --mode T --config_path configs/${model}-${dataset}.ini --out_dir /data/lastness/output/${model}-${dataset}
#    done
#  done

for dataset in tweet sina
do
  for model in RNN
  do
  python main.py --mode E --config_path configs/${model}-${dataset}-adgv2.ini --out_dir /data/lastness/output/${model}-${dataset}
  done
  done
