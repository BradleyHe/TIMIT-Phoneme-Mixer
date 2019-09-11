if [ "$#" -ne 1 ]; then
    echo "Usage : ./timit_preprocess.sh <timit folder>"
fi
echo 'Transfering raw TIMIT wave file format from NIST to RIFF.'
echo ' '
find $1 -name '*.WAV' | parallel -P20 sox {} '{.}.wav'
echo 'Done'