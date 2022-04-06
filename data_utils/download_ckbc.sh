mkdir data/
mkdir data/ckbc/

wget https://home.ttic.edu/~kgimpel/comsense_resources/test.txt.gz
gzip -d test.txt.gz

mv test.txt data/ckbc/
