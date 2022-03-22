mkdir data/
cd data/

wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
rm data.zip

mv data/ lama/