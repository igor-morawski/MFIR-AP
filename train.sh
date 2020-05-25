(. ./setups.sh

for setup in "${setups[@]}"; do 
    python3 train.py $setup
done
echo "train.sh executed"

)