# create res directory if it doesnt exist
if [ ! -d "res" ]; then
  mkdir res
fi

#! download the data, insert new file names at &files=<FILE_NAME> as applicable
# training & full https://drive.switch.ch/index.php/s/rPCEnAHyTrXVAY1
wget https://drive.switch.ch/index.php/s/rPCEnAHyTrXVAY1/download?path=/\&files=english_full.txt -O res/english_full.txt
wget https://drive.switch.ch/index.php/s/rPCEnAHyTrXVAY1/download?path=/\&files=english_train.txt -O res/english_train.txt

# validation https://drive.switch.ch/index.php/s/Lau2Y4vGgds8wtu
wget https://drive.switch.ch/index.php/s/Lau2Y4vGgds8wtu/download?path=/\&files=english_valid.txt -O res/english_valid.txt

# test https://drive.switch.ch/index.php/s/8HVSN2d2KIwffDR
wget https://drive.switch.ch/index.php/s/8HVSN2d2KIwffDR/download?path=/\&files=sango.txt -O res/sango_test.txt
