(. ./setups.sh

for setup in "${setups[@]}"; do 
    python3 test.py $setup
done
echo "test.sh executed"

)