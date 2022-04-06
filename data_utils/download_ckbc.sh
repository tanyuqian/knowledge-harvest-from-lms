mkdir data/
mkdir data/ckbc/

wget https://home.ttic.edu/~kgimpel/comsense_resources/test.txt.gz --no-check-certificate
gzip -d test.txt.gz

mv test.txt data/ckbc/
mkdir ckbc_curves/