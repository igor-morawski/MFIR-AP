#train-test-compare
(. ./setups.sh

for setup in "${setups[@]}"; do 
    python3 train.py $setup
    python3 test.py $setup
    python3 report.py $setup
done
echo "training and testing (incl. reporting) executed"
python3 compare.py "${setups[@]}" "${completed_setups[@]}" 
echo "compare.py executed"
)