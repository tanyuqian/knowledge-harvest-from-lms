wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gzip -d conceptnet-assertions-5.7.0.csv.gz
python data_utils/build_conceptnet_dataset.sh
