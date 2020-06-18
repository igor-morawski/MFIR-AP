(. ./setups.sh

for setup in "${setups[@]}"; do 
    python3 report.py $setup
    echo "reported $setup" 
done
for setup in "${completed_setups[@]}"; do 
    python3 report.py $setup
done
python3 compare.py "${setups[@]}" "${completed_setups[@]}" 
echo "compare.sh executed"

)