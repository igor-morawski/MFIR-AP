#train-test-compare
(. ./setups.sh

for setup in "${setups[@]}"; do 
    python3 train.py $setup
    python3 test.py $setup
done
echo "training and testing executed"
python3 compare.py "${setups[@]}" "${completed_setups[@]}" 
echo "compare.py executed"
)