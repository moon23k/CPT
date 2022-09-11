#!/bin/bash
mkdir -p data
cd data

wget http://yanran.li/files/ijcnlp_dailydialog.zip
unzip *.zip
rm *.zip

mv ijcnlp_dailydialog/dialogues_text.txt .
rm -rf ijcnlp_dailydialog

python3 process_data.py